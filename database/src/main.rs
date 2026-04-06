use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use common::{
    Data, DataType,
    query::{
        ComparisionOperator, ComparisionValue, FilterData, Predicate, ProjectData, Query,
        QueryOp, ScanData, SortData,
    },
};
use db_config::{DbContext, table::TableSpec};
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    rc::Rc,
};

use crate::{
    cli::CliOptions,
    io_setup::{setup_disk_io, setup_monitor_io},
};

mod cli;
mod io_setup;

#[derive(Clone)]
struct ColumnMeta {
    name: String,
    data_type: DataType,
}

#[derive(Clone)]
struct Row {
    values: Vec<Data>,
}

struct ResultSet {
    columns: Vec<ColumnMeta>,
    rows: Vec<Row>,
}

trait RowSource {
    fn columns(&self) -> &[ColumnMeta];
    fn next_row(&mut self) -> Result<Option<Row>>;
}

type SharedDiskClient = Rc<RefCell<DiskClient>>;
const SCAN_BATCH_BLOCKS: u64 = 128;

struct DiskClient {
    reader: BufReader<Box<dyn Read>>,
    writer: Box<dyn Write>,
    block_size: usize,
}

impl DiskClient {
    fn new(mut reader: BufReader<Box<dyn Read>>, mut writer: Box<dyn Write>) -> Result<Self> {
        writer.write_all(b"get block-size\n")?;
        writer.flush()?;

        let block_size = read_u64_line(&mut reader)? as usize;

        Ok(Self {
            reader,
            writer,
            block_size,
        })
    }

    fn get_file_start_block(&mut self, file_id: &str) -> Result<u64> {
        self.writer
            .write_all(format!("get file start-block {file_id}\n").as_bytes())?;
        self.writer.flush()?;
        read_u64_line(&mut self.reader)
    }

    fn get_file_num_blocks(&mut self, file_id: &str) -> Result<u64> {
        self.writer
            .write_all(format!("get file num-blocks {file_id}\n").as_bytes())?;
        self.writer.flush()?;
        read_u64_line(&mut self.reader)
    }

    fn get_blocks(&mut self, start_block_id: u64, num_blocks: u64) -> Result<Vec<u8>> {
        self.writer
            .write_all(format!("get block {start_block_id} {num_blocks}\n").as_bytes())?;
        self.writer.flush()?;

        let mut bytes = vec![0u8; (num_blocks as usize) * self.block_size];
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

struct MaterializedSource {
    columns: Vec<ColumnMeta>,
    rows: std::vec::IntoIter<Row>,
}

impl MaterializedSource {
    fn new(result: ResultSet) -> Self {
        Self {
            columns: result.columns,
            rows: result.rows.into_iter(),
        }
    }
}

impl RowSource for MaterializedSource {
    fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Row>> {
        Ok(self.rows.next())
    }
}

struct ScanSource {
    columns: Vec<ColumnMeta>,
    scan_layout: Vec<ScanValueSpec>,
    disk_client: SharedDiskClient,
    start_block: u64,
    num_blocks: u64,
    next_block_offset: u64,
    current_rows: std::vec::IntoIter<Row>,
}

#[derive(Clone)]
struct ScanValueSpec {
    data_type: DataType,
    output_index: Option<usize>,
}

impl ScanSource {
    fn new(
        table: &TableSpec,
        disk_client: SharedDiskClient,
        required_columns: Option<&HashSet<String>>,
    ) -> Result<Self> {
        let (start_block, num_blocks, columns, scan_layout) = {
            let mut disk = disk_client.borrow_mut();
            let start_block = disk.get_file_start_block(&table.file_id)?;
            let num_blocks = disk.get_file_num_blocks(&table.file_id)?;
            let (columns, scan_layout) = build_scan_plan(table, required_columns);

            (start_block, num_blocks, columns, scan_layout)
        };

        Ok(Self {
            columns,
            scan_layout,
            disk_client,
            start_block,
            num_blocks,
            next_block_offset: 0,
            current_rows: Vec::new().into_iter(),
        })
    }

    fn load_next_block_rows(&mut self) -> Result<bool> {
        if self.next_block_offset >= self.num_blocks {
            return Ok(false);
        }

        let blocks_to_read = (self.num_blocks - self.next_block_offset).min(SCAN_BATCH_BLOCKS);

        let (block_size, batch_bytes) = {
            let mut disk = self.disk_client.borrow_mut();
            let block_size = disk.block_size;
            let batch_bytes =
                disk.get_blocks(self.start_block + self.next_block_offset, blocks_to_read)?;
            (block_size, batch_bytes)
        };

        self.next_block_offset += blocks_to_read;
        let rows = decode_batch_rows(&self.scan_layout, &batch_bytes, block_size)?;
        self.current_rows = rows.into_iter();
        Ok(true)
    }
}

fn build_scan_plan(
    table: &TableSpec,
    required_columns: Option<&HashSet<String>>,
) -> (Vec<ColumnMeta>, Vec<ScanValueSpec>) {
    let mut columns = Vec::new();
    let mut scan_layout = Vec::with_capacity(table.column_specs.len());

    for column in &table.column_specs {
        let include = required_columns
            .map(|needed| needed.contains(column.column_name.as_str()))
            .unwrap_or(true);

        let output_index = if include {
            let idx = columns.len();
            columns.push(ColumnMeta {
                name: column.column_name.clone(),
                data_type: column.data_type.clone(),
            });
            Some(idx)
        } else {
            None
        };

        scan_layout.push(ScanValueSpec {
            data_type: column.data_type.clone(),
            output_index,
        });
    }

    (columns, scan_layout)
}

impl RowSource for ScanSource {
    fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Row>> {
        loop {
            if let Some(row) = self.current_rows.next() {
                return Ok(Some(row));
            }

            if !self.load_next_block_rows()? {
                return Ok(None);
            }
        }
    }
}

struct FilterSource {
    columns: Vec<ColumnMeta>,
    predicates: Vec<Predicate>,
    column_index: HashMap<String, usize>,
    child: Box<dyn RowSource>,
}

impl FilterSource {
    fn new(filter: &FilterData, child: Box<dyn RowSource>) -> Self {
        let columns = child.columns().to_vec();
        let column_index = build_column_index(&columns);

        Self {
            columns,
            predicates: filter.predicates.clone(),
            column_index,
            child,
        }
    }
}

impl RowSource for FilterSource {
    fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Row>> {
        while let Some(row) = self.child.next_row()? {
            let mut keep_row = true;

            for predicate in &self.predicates {
                if !predicate_matches(predicate, &row, &self.column_index)? {
                    keep_row = false;
                    break;
                }
            }

            if keep_row {
                return Ok(Some(row));
            }
        }

        Ok(None)
    }
}

struct ProjectSource {
    columns: Vec<ColumnMeta>,
    projection: Vec<usize>,
    child: Box<dyn RowSource>,
}

#[derive(Clone)]
struct CompiledPredicate {
    left_idx: usize,
    operator: ComparisionOperator,
    right: CompiledPredicateValue,
}

#[derive(Clone)]
enum CompiledPredicateValue {
    Column(usize),
    Literal(Data),
}

struct FilterProjectScanSource {
    columns: Vec<ColumnMeta>,
    projection: Vec<usize>,
    predicates: Vec<CompiledPredicate>,
    scan: ScanSource,
}

#[derive(Clone)]
struct FastProjectColumn {
    source_idx: usize,
}

impl FilterProjectScanSource {
    fn new(project: &ProjectData, filter: &FilterData, scan: ScanSource) -> Result<Self> {
        let scan_columns = scan.columns().to_vec();
        let column_index = build_column_index(&scan_columns);
        let projection = project
            .column_name_map
            .iter()
            .map(|(from, _)| {
                column_index
                    .get(from.as_str())
                    .copied()
                    .ok_or_else(|| anyhow!("Unknown project column {from}"))
            })
            .collect::<Result<Vec<_>>>()?;

        let columns = project
            .column_name_map
            .iter()
            .map(|(from, to)| {
                let idx = *column_index
                    .get(from.as_str())
                    .ok_or_else(|| anyhow!("Unknown project column {from}"))?;
                Ok(ColumnMeta {
                    name: to.clone(),
                    data_type: scan_columns[idx].data_type.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let predicates = filter
            .predicates
            .iter()
            .map(|predicate| compile_predicate(predicate, &column_index))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            columns,
            projection,
            predicates,
            scan,
        })
    }
}

impl RowSource for FilterProjectScanSource {
    fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Row>> {
        while let Some(row) = self.scan.next_row()? {
            let mut keep = true;
            for predicate in &self.predicates {
                if !compiled_predicate_matches(predicate, &row)? {
                    keep = false;
                    break;
                }
            }

            if keep {
                return Ok(Some(Row {
                    values: self
                        .projection
                        .iter()
                        .map(|idx| row.values[*idx].clone())
                        .collect(),
                }));
            }
        }

        Ok(None)
    }
}

impl ProjectSource {
    fn new(project: &ProjectData, child: Box<dyn RowSource>) -> Result<Self> {
        let child_columns = child.columns().to_vec();
        let child_index = build_column_index(&child_columns);

        let projection_data = project
            .column_name_map
            .iter()
            .map(|(from, to)| {
                let idx = *child_index
                    .get(from.as_str())
                    .ok_or_else(|| anyhow!("Unknown project column {from}"))?;
                Ok((idx, to.clone(), child_columns[idx].data_type.clone()))
            })
            .collect::<Result<Vec<_>>>()?;

        let projection = projection_data.iter().map(|(idx, _, _)| *idx).collect();
        let columns = projection_data
            .into_iter()
            .map(|(_, name, data_type)| ColumnMeta { name, data_type })
            .collect();

        Ok(Self {
            columns,
            projection,
            child,
        })
    }
}

impl RowSource for ProjectSource {
    fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Row>> {
        let Some(row) = self.child.next_row()? else {
            return Ok(None);
        };

        Ok(Some(Row {
            values: self
                .projection
                .iter()
                .map(|idx| row.values[*idx].clone())
                .collect(),
        }))
    }
}

fn read_u64_line(reader: &mut impl BufRead) -> Result<u64> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    line.trim()
        .parse()
        .with_context(|| format!("Failed to parse integer response: {}", line.trim()))
}

fn find_table<'a>(ctx: &'a DbContext, table_name: &str) -> Result<&'a TableSpec> {
    ctx.get_table_specs()
        .iter()
        .find(|table| table.name == table_name)
        .ok_or_else(|| anyhow!("Unknown table {table_name}"))
}

fn build_column_index(columns: &[ColumnMeta]) -> HashMap<String, usize> {
    columns
        .iter()
        .enumerate()
        .map(|(idx, column)| (column.name.clone(), idx))
        .collect()
}

fn decode_value(data_type: &DataType, buf: &[u8], offset: &mut usize) -> Result<Data> {
    match data_type {
        DataType::Int32 => {
            let end = *offset + 4;
            let bytes: [u8; 4] = buf
                .get(*offset..end)
                .ok_or_else(|| anyhow!("Int32 exceeded block boundary"))?
                .try_into()
                .map_err(|_| anyhow!("Invalid Int32 bytes"))?;
            *offset = end;
            Ok(Data::Int32(i32::from_le_bytes(bytes)))
        }
        DataType::Int64 => {
            let end = *offset + 8;
            let bytes: [u8; 8] = buf
                .get(*offset..end)
                .ok_or_else(|| anyhow!("Int64 exceeded block boundary"))?
                .try_into()
                .map_err(|_| anyhow!("Invalid Int64 bytes"))?;
            *offset = end;
            Ok(Data::Int64(i64::from_le_bytes(bytes)))
        }
        DataType::Float32 => {
            let end = *offset + 4;
            let bytes: [u8; 4] = buf
                .get(*offset..end)
                .ok_or_else(|| anyhow!("Float32 exceeded block boundary"))?
                .try_into()
                .map_err(|_| anyhow!("Invalid Float32 bytes"))?;
            *offset = end;
            Ok(Data::Float32(f32::from_le_bytes(bytes)))
        }
        DataType::Float64 => {
            let end = *offset + 8;
            let bytes: [u8; 8] = buf
                .get(*offset..end)
                .ok_or_else(|| anyhow!("Float64 exceeded block boundary"))?
                .try_into()
                .map_err(|_| anyhow!("Invalid Float64 bytes"))?;
            *offset = end;
            Ok(Data::Float64(f64::from_le_bytes(bytes)))
        }
        DataType::String => {
            let tail = buf
                .get(*offset..)
                .ok_or_else(|| anyhow!("String exceeded block boundary"))?;
            let Some(len) = tail.iter().position(|byte| *byte == 0) else {
                bail!("String terminator not found inside block");
            };
            let string = String::from_utf8(tail[..len].to_vec()).context("Invalid UTF-8 data")?;
            *offset += len + 1;
            Ok(Data::String(string))
        }
    }
}

fn decode_value_or_skip(
    value_spec: &ScanValueSpec,
    bytes: &[u8],
    offset: &mut usize,
) -> Result<Option<Data>> {
    match (&value_spec.data_type, value_spec.output_index) {
        (DataType::Int32, Some(_)) => decode_value(&DataType::Int32, bytes, offset).map(Some),
        (DataType::Int64, Some(_)) => decode_value(&DataType::Int64, bytes, offset).map(Some),
        (DataType::Float32, Some(_)) => decode_value(&DataType::Float32, bytes, offset).map(Some),
        (DataType::Float64, Some(_)) => decode_value(&DataType::Float64, bytes, offset).map(Some),
        (DataType::String, Some(_)) => decode_value(&DataType::String, bytes, offset).map(Some),
        (DataType::Int32, None) => {
            *offset += 4;
            Ok(None)
        }
        (DataType::Int64, None) => {
            *offset += 8;
            Ok(None)
        }
        (DataType::Float32, None) => {
            *offset += 4;
            Ok(None)
        }
        (DataType::Float64, None) => {
            *offset += 8;
            Ok(None)
        }
        (DataType::String, None) => {
            let tail = bytes
                .get(*offset..)
                .ok_or_else(|| anyhow!("String exceeded block boundary"))?;
            let Some(len) = tail.iter().position(|byte| *byte == 0) else {
                bail!("String terminator not found inside block");
            };
            *offset += len + 1;
            Ok(None)
        }
    }
}

fn decode_block_rows(
    scan_layout: &[ScanValueSpec],
    bytes: &[u8],
    block_size: usize,
) -> Result<Vec<Row>> {
    if bytes.len() != block_size {
        bail!(
            "Expected exactly one block of size {block_size}, got {} bytes",
            bytes.len()
        );
    }

    let num_output_columns = scan_layout
        .iter()
        .filter(|spec| spec.output_index.is_some())
        .count();
    let row_count_offset = block_size - 2;
    let row_count = u16::from_le_bytes([bytes[row_count_offset], bytes[row_count_offset + 1]]);
    let mut offset = 0usize;
    let payload = &bytes[..row_count_offset];
    let mut rows = Vec::with_capacity(row_count as usize);

    for _ in 0..row_count {
        let mut values = vec![Data::Int32(0); num_output_columns];
        for value_spec in scan_layout {
            if let Some(value) = decode_value_or_skip(value_spec, payload, &mut offset)? {
                values[value_spec.output_index.unwrap()] = value;
            }
        }
        rows.push(Row { values });
    }

    Ok(rows)
}

fn decode_batch_rows(
    scan_layout: &[ScanValueSpec],
    bytes: &[u8],
    block_size: usize,
) -> Result<Vec<Row>> {
    if bytes.len() % block_size != 0 {
        bail!(
            "Batch length {} is not a multiple of block size {block_size}",
            bytes.len()
        );
    }

    let mut rows = Vec::new();
    for block in bytes.chunks_exact(block_size) {
        rows.extend(decode_block_rows(scan_layout, block, block_size)?);
    }

    Ok(rows)
}

fn collect_required_columns_for_filter(
    filter: &FilterData,
    required_columns: &mut HashSet<String>,
) {
    for predicate in &filter.predicates {
        required_columns.insert(predicate.column_name.clone());
        if let ComparisionValue::Column(name) = &predicate.value {
            required_columns.insert(name.clone());
        }
    }
}

fn comparison_value_to_data(
    value: &ComparisionValue,
    row: &Row,
    column_index: &HashMap<String, usize>,
) -> Result<Data> {
    match value {
        ComparisionValue::Column(name) => {
            let idx = *column_index
                .get(name.as_str())
                .ok_or_else(|| anyhow!("Unknown predicate column {name}"))?;
            Ok(row.values[idx].clone())
        }
        ComparisionValue::I32(value) => Ok(Data::Int32(*value)),
        ComparisionValue::I64(value) => Ok(Data::Int64(*value)),
        ComparisionValue::F32(value) => Ok(Data::Float32(*value)),
        ComparisionValue::F64(value) => Ok(Data::Float64(*value)),
        ComparisionValue::String(value) => Ok(Data::String(value.clone())),
    }
}

fn compile_predicate(
    predicate: &Predicate,
    column_index: &HashMap<String, usize>,
) -> Result<CompiledPredicate> {
    let left_idx = *column_index
        .get(predicate.column_name.as_str())
        .ok_or_else(|| anyhow!("Unknown filter column {}", predicate.column_name))?;
    let right = match &predicate.value {
        ComparisionValue::Column(name) => CompiledPredicateValue::Column(
            *column_index
                .get(name.as_str())
                .ok_or_else(|| anyhow!("Unknown predicate column {name}"))?,
        ),
        ComparisionValue::I32(value) => CompiledPredicateValue::Literal(Data::Int32(*value)),
        ComparisionValue::I64(value) => CompiledPredicateValue::Literal(Data::Int64(*value)),
        ComparisionValue::F32(value) => CompiledPredicateValue::Literal(Data::Float32(*value)),
        ComparisionValue::F64(value) => CompiledPredicateValue::Literal(Data::Float64(*value)),
        ComparisionValue::String(value) => {
            CompiledPredicateValue::Literal(Data::String(value.clone()))
        }
    };

    Ok(CompiledPredicate {
        left_idx,
        operator: predicate.operator.clone(),
        right,
    })
}

fn compiled_predicate_matches(predicate: &CompiledPredicate, row: &Row) -> Result<bool> {
    let left = &row.values[predicate.left_idx];
    let right = match &predicate.right {
        CompiledPredicateValue::Column(idx) => row.values[*idx].clone(),
        CompiledPredicateValue::Literal(value) => value.clone(),
    };

    Ok(compare_data(left, &predicate.operator, &right))
}

fn data_to_output(data: &Data) -> String {
    match data {
        Data::Int32(value) => value.to_string(),
        Data::Int64(value) => value.to_string(),
        Data::Float32(value) => {
            let s = value.to_string();
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{}.0", s)
            }
        }
        Data::Float64(value) => {
            let s = value.to_string();
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{}.0", s)
            }
        }
        Data::String(value) => value.clone(),
    }
}

fn write_project_filter_scan_fast(
    project: &ProjectData,
    filter: &FilterData,
    table: &TableSpec,
    disk_client: SharedDiskClient,
    monitor_out: &mut dyn Write,
) -> Result<()> {
    let mut required_columns = project
        .column_name_map
        .iter()
        .map(|(from, _)| from.clone())
        .collect::<HashSet<_>>();
    collect_required_columns_for_filter(filter, &mut required_columns);

    let (scan_columns, scan_layout) = build_scan_plan(table, Some(&required_columns));
    let column_index = build_column_index(&scan_columns);
    let predicates = filter
        .predicates
        .iter()
        .map(|predicate| compile_predicate(predicate, &column_index))
        .collect::<Result<Vec<_>>>()?;
    let projection = project
        .column_name_map
        .iter()
        .map(|(from, _)| {
            Ok(FastProjectColumn {
                source_idx: *column_index
                    .get(from.as_str())
                    .ok_or_else(|| anyhow!("Unknown project column {from}"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let (start_block, num_blocks, block_size) = {
        let mut disk = disk_client.borrow_mut();
        (
            disk.get_file_start_block(&table.file_id)?,
            disk.get_file_num_blocks(&table.file_id)?,
            disk.block_size,
        )
    };

    monitor_out.write_all(b"validate\n")?;
    monitor_out.flush()?;

    let mut line = String::new();
    let num_output_columns = scan_columns.len();
    let mut next_block_offset = 0u64;

    while next_block_offset < num_blocks {
        let blocks_to_read = (num_blocks - next_block_offset).min(SCAN_BATCH_BLOCKS);
        let batch_bytes = {
            let mut disk = disk_client.borrow_mut();
            disk.get_blocks(start_block + next_block_offset, blocks_to_read)?
        };
        next_block_offset += blocks_to_read;

        for block in batch_bytes.chunks_exact(block_size) {
            let row_count_offset = block_size - 2;
            let row_count = u16::from_le_bytes([block[row_count_offset], block[row_count_offset + 1]]);
            let payload = &block[..row_count_offset];
            let mut offset = 0usize;

            for _ in 0..row_count {
                let mut values = Vec::with_capacity(num_output_columns);
                for value_spec in &scan_layout {
                    if let Some(value) = decode_value_or_skip(value_spec, payload, &mut offset)? {
                        values.push(value);
                    }
                }

                let row = Row { values };
                let mut keep = true;
                for predicate in &predicates {
                    if !compiled_predicate_matches(predicate, &row)? {
                        keep = false;
                        break;
                    }
                }

                if keep {
                    line.clear();
                    for project_column in &projection {
                        line.push_str(&data_to_output(&row.values[project_column.source_idx]));
                        line.push('|');
                    }
                    line.push('\n');
                    monitor_out.write_all(line.as_bytes())?;
                }
            }
        }
    }

    monitor_out.write_all(b"!\n")?;
    monitor_out.flush()?;
    Ok(())
}

fn compare_data(left: &Data, operator: &ComparisionOperator, right: &Data) -> bool {
    match operator {
        ComparisionOperator::EQ => left == right,
        ComparisionOperator::NE => left != right,
        ComparisionOperator::GT => left.partial_cmp(right) == Some(Ordering::Greater),
        ComparisionOperator::GTE => {
            matches!(
                left.partial_cmp(right),
                Some(Ordering::Greater) | Some(Ordering::Equal)
            )
        }
        ComparisionOperator::LT => left.partial_cmp(right) == Some(Ordering::Less),
        ComparisionOperator::LTE => {
            matches!(
                left.partial_cmp(right),
                Some(Ordering::Less) | Some(Ordering::Equal)
            )
        }
    }
}

fn predicate_matches(
    predicate: &Predicate,
    row: &Row,
    column_index: &HashMap<String, usize>,
) -> Result<bool> {
    let left_idx = *column_index
        .get(predicate.column_name.as_str())
        .ok_or_else(|| anyhow!("Unknown filter column {}", predicate.column_name))?;
    let left = &row.values[left_idx];
    let right = comparison_value_to_data(&predicate.value, row, column_index)?;
    Ok(compare_data(left, &predicate.operator, &right))
}

fn sort_rows(sort: &SortData, rows: &mut [Row], columns: &[ColumnMeta]) -> Result<()> {
    let column_index = build_column_index(columns);
    let sort_columns = sort
        .sort_specs
        .iter()
        .map(|spec| {
            let idx = *column_index
                .get(spec.column_name.as_str())
                .ok_or_else(|| anyhow!("Unknown sort column {}", spec.column_name))?;
            Ok((idx, spec.ascending))
        })
        .collect::<Result<Vec<_>>>()?;

    rows.sort_by(|left, right| {
        for (idx, ascending) in &sort_columns {
            let ordering = left.values[*idx]
                .partial_cmp(&right.values[*idx])
                .unwrap_or(Ordering::Equal);

            if ordering != Ordering::Equal {
                return if *ascending {
                    ordering
                } else {
                    ordering.reverse()
                };
            }
        }

        Ordering::Equal
    });

    Ok(())
}

fn execute_cross(left: ResultSet, right: ResultSet) -> ResultSet {
    let mut columns = left.columns;
    columns.extend(right.columns);

    let mut rows = Vec::with_capacity(left.rows.len().saturating_mul(right.rows.len()));

    for left_row in left.rows {
        for right_row in &right.rows {
            let mut values = left_row.values.clone();
            values.extend(right_row.values.clone());
            rows.push(Row { values });
        }
    }

    ResultSet { columns, rows }
}

fn materialize_source(mut source: Box<dyn RowSource>) -> Result<ResultSet> {
    let columns = source.columns().to_vec();
    let mut rows = Vec::new();

    while let Some(row) = source.next_row()? {
        rows.push(row);
    }

    Ok(ResultSet { columns, rows })
}

fn execute_query_op(
    query_op: &QueryOp,
    ctx: &DbContext,
    disk_client: SharedDiskClient,
) -> Result<Box<dyn RowSource>> {
    match query_op {
        QueryOp::Scan(ScanData { table_id }) => {
            let table = find_table(ctx, table_id)?;
            Ok(Box::new(ScanSource::new(table, disk_client, None)?))
        }
        QueryOp::Filter(filter) => {
            let child = execute_query_op(&filter.underlying, ctx, disk_client)?;
            Ok(Box::new(FilterSource::new(filter, child)))
        }
        QueryOp::Project(project) => {
            if let QueryOp::Scan(ScanData { table_id }) = project.underlying.as_ref() {
                let table = find_table(ctx, table_id)?;
                let required_columns = project
                    .column_name_map
                    .iter()
                    .map(|(from, _)| from.clone())
                    .collect::<HashSet<_>>();
                let scan = ScanSource::new(table, disk_client, Some(&required_columns))?;
                return Ok(Box::new(ProjectSource::new(project, Box::new(scan))?));
            }

            if let QueryOp::Filter(filter) = project.underlying.as_ref() {
                if let QueryOp::Scan(ScanData { table_id }) = filter.underlying.as_ref() {
                    let table = find_table(ctx, table_id)?;
                    let mut required_columns = project
                        .column_name_map
                        .iter()
                        .map(|(from, _)| from.clone())
                        .collect::<HashSet<_>>();
                    collect_required_columns_for_filter(filter, &mut required_columns);

                    let scan = ScanSource::new(table, disk_client, Some(&required_columns))?;
                    return Ok(Box::new(FilterProjectScanSource::new(
                        project, filter, scan,
                    )?));
                }
            }

            let child = execute_query_op(&project.underlying, ctx, disk_client)?;
            Ok(Box::new(ProjectSource::new(project, child)?))
        }
        QueryOp::Sort(sort) => {
            let child = execute_query_op(&sort.underlying, ctx, disk_client)?;
            let mut result = materialize_source(child)?;
            sort_rows(sort, &mut result.rows, &result.columns)?;
            Ok(Box::new(MaterializedSource::new(result)))
        }
        QueryOp::Cross(cross) => {
            let left = materialize_source(execute_query_op(&cross.left, ctx, disk_client.clone())?)?;
            let right = materialize_source(execute_query_op(&cross.right, ctx, disk_client)?)?;
            Ok(Box::new(MaterializedSource::new(execute_cross(left, right))))
        }
    }
}

fn write_result_to_monitor(
    mut result_source: Box<dyn RowSource>,
    monitor_out: &mut dyn Write,
) -> Result<()> {
    monitor_out.write_all(b"validate\n")?;

    while let Some(row) = result_source.next_row()? {
        let mut line = String::new();
        for value in &row.values {
            line.push_str(&data_to_output(value));
            line.push('|');
        }
        line.push('\n');
        monitor_out.write_all(line.as_bytes())?;
    }

    monitor_out.write_all(b"!\n")?;
    monitor_out.flush()?;
    Ok(())
}

fn read_query(monitor_in: &mut impl BufRead) -> Result<Query> {
    let mut input_line = String::new();
    monitor_in.read_line(&mut input_line)?;
    serde_json::from_str(&input_line).context("Failed to parse query from monitor")
}

fn request_memory_limit(
    monitor_in: &mut impl BufRead,
    monitor_out: &mut impl Write,
) -> Result<u64> {
    monitor_out.write_all(b"get_memory_limit\n")?;
    monitor_out.flush()?;
    read_u64_line(monitor_in)
}

fn db_main() -> Result<()> {
    let cli_options = CliOptions::parse();
    let ctx = DbContext::load_from_file(cli_options.get_config_path())?;

    let (disk_in, disk_out) = setup_disk_io();
    let (monitor_in, monitor_out) = setup_monitor_io();

    let disk_reader: Box<dyn Read> = Box::new(disk_in);
    let disk_writer: Box<dyn Write> = Box::new(disk_out);
    let mut monitor_reader = BufReader::new(monitor_in);
    let mut monitor_writer = BufWriter::new(monitor_out);

    let disk_client = Rc::new(RefCell::new(DiskClient::new(
        BufReader::new(disk_reader),
        disk_writer,
    )?));

    let query = read_query(&mut monitor_reader)?;
    let _memory_limit_mb = request_memory_limit(&mut monitor_reader, &mut monitor_writer)?;

    if let QueryOp::Project(project) = &query.root {
        if let QueryOp::Filter(filter) = project.underlying.as_ref() {
            if let QueryOp::Scan(ScanData { table_id }) = filter.underlying.as_ref() {
                let table = find_table(&ctx, table_id)?;
                return write_project_filter_scan_fast(
                    project,
                    filter,
                    table,
                    disk_client,
                    &mut monitor_writer,
                );
            }
        }
    }

    let result_source = execute_query_op(&query.root, &ctx, disk_client)?;
    write_result_to_monitor(result_source, &mut monitor_writer)?;

    Ok(())
}

fn main() -> Result<()> {
    db_main().with_context(|| "From Database")
}
