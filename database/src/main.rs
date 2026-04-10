use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use common::{
    Data, DataType,
    query::{
        ComparisionOperator, ComparisionValue, CrossData, FilterData, Predicate, ProjectData, Query,
        QueryOp, ScanData, SortData,
    },
};
use db_config::{DbContext, table::TableSpec};
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    rc::Rc,
};

use crate::{
    cli::CliOptions,
    io_setup::{setup_disk_io, setup_monitor_io},
};

use scratch::ScratchAllocator;      //APal

mod cli;
mod io_setup;
mod scratch;                        //APal

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

    fn get_anon_start_block(&mut self) -> Result<u64> {
        self.writer.write_all(b"get anon-start-block\n")?;
        self.writer.flush()?;
        read_u64_line(&mut self.reader)
    }

    fn put_blocks(&mut self, start_block_id: u64, data: &[u8]) -> Result<()> {
        let num_blocks = data.len() / self.block_size;
        self.writer.write_all(
            format!("put block {start_block_id} {num_blocks}\n").as_bytes()
        )?;
        self.writer.write_all(data)?;
        self.writer.flush()?;
        Ok(())
    }
}

// ── Scratch-space row serialization ─────────────────────────────────────────

/// Wrapper around Data that implements Hash + Eq for use as a HashMap key.
struct DataKey(Data);

impl Hash for DataKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self.0 {
            Data::Int32(v)  => { 0u8.hash(state); v.hash(state); }
            Data::Int64(v)  => { 1u8.hash(state); v.hash(state); }
            Data::Float32(v) => { 2u8.hash(state); v.to_bits().hash(state); }
            Data::Float64(v) => { 3u8.hash(state); v.to_bits().hash(state); }
            Data::String(v) => { 4u8.hash(state); v.hash(state); }
        }
    }
}

impl PartialEq for DataKey {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}
impl Eq for DataKey {}

/// Encode a single Data value into a byte buffer.
/// Strings: 2-byte LE length prefix + raw UTF-8 bytes.
fn encode_scratch_value(data: &Data, buf: &mut Vec<u8>) {
    match data {
        Data::Int32(v)  => buf.extend_from_slice(&v.to_le_bytes()),
        Data::Int64(v)  => buf.extend_from_slice(&v.to_le_bytes()),
        Data::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Data::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Data::String(s) => {
            let bytes = s.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
    }
}

/// Exact encoded byte size for a Data value.
fn scratch_value_size(data: &Data) -> usize {
    match data {
        Data::Int32(_)  => 4,
        Data::Int64(_)  => 8,
        Data::Float32(_) => 4,
        Data::Float64(_) => 8,
        Data::String(s) => 2 + s.len(),
    }
}

/// Realistic in-memory per-row size estimate covering Vecs and Enums.
fn scratch_row_size_estimate(schema: &[DataType]) -> usize {
    // True memory overhead in Rust per row:
    // Row struct = 24 bytes (for Vec<Data>)
    // Vec allocation: schema.len() * 32 bytes
    let mut size = 24 + schema.len() * 32;

    for dt in schema {
        if matches!(dt, DataType::String) {
            size += 64; // conservative average string payload overhead on heap
        }
    }
    size.max(1)
}

/// Decode one Data value from a byte slice at *offset (advances offset).
fn decode_scratch_value(dtype: &DataType, buf: &[u8], offset: &mut usize) -> Result<Data> {
    match dtype {
        DataType::Int32 => {
            let b: [u8; 4] = buf[*offset..*offset+4].try_into()
                .map_err(|_| anyhow!("scratch Int32 OOB"))?;
            *offset += 4;
            Ok(Data::Int32(i32::from_le_bytes(b)))
        }
        DataType::Int64 => {
            let b: [u8; 8] = buf[*offset..*offset+8].try_into()
                .map_err(|_| anyhow!("scratch Int64 OOB"))?;
            *offset += 8;
            Ok(Data::Int64(i64::from_le_bytes(b)))
        }
        DataType::Float32 => {
            let b: [u8; 4] = buf[*offset..*offset+4].try_into()
                .map_err(|_| anyhow!("scratch Float32 OOB"))?;
            *offset += 4;
            Ok(Data::Float32(f32::from_le_bytes(b)))
        }
        DataType::Float64 => {
            let b: [u8; 8] = buf[*offset..*offset+8].try_into()
                .map_err(|_| anyhow!("scratch Float64 OOB"))?;
            *offset += 8;
            Ok(Data::Float64(f64::from_le_bytes(b)))
        }
        DataType::String => {
            let len_b: [u8; 2] = buf[*offset..*offset+2].try_into()
                .map_err(|_| anyhow!("scratch String len OOB"))?;
            let len = u16::from_le_bytes(len_b) as usize;
            *offset += 2;
            let s = String::from_utf8(buf[*offset..*offset+len].to_vec())
                .context("scratch String UTF-8")?;
            *offset += len;
            Ok(Data::String(s))
        }
    }
}

/// Pack rows into scratch blocks (each block_size bytes, last 2 bytes = u16 row count LE).
/// Returns a Vec<u8> whose length is a multiple of block_size.
fn encode_rows_to_scratch_blocks(rows: &[Row], _schema: &[DataType], block_size: usize) -> Vec<u8> {
    let mut all = Vec::new();
    let mut block = vec![0u8; block_size];
    let mut offset = 0usize;
    let mut count = 0u16;

    for row in rows {
        // Encode the row into a temporary buffer
        let mut rbuf = Vec::new();
        for val in &row.values {
            encode_scratch_value(val, &mut rbuf);
        }
        if rbuf.len() > block_size - 2 {
            // Row too large for any block — skip (should not happen with TPCH data)
            continue;
        }
        if offset + rbuf.len() > block_size - 2 {
            // Flush current block
            let cb = count.to_le_bytes();
            block[block_size - 2] = cb[0];
            block[block_size - 1] = cb[1];
            all.extend_from_slice(&block);
            block = vec![0u8; block_size];
            offset = 0;
            count = 0;
        }
        block[offset..offset + rbuf.len()].copy_from_slice(&rbuf);
        offset += rbuf.len();
        count += 1;
    }
    // Flush last block (may be empty if rows was empty)
    let cb = count.to_le_bytes();
    block[block_size - 2] = cb[0];
    block[block_size - 1] = cb[1];
    all.extend_from_slice(&block);
    all
}

/// Decode all rows from one scratch block.
fn decode_scratch_block(block: &[u8], schema: &[DataType]) -> Result<Vec<Row>> {
    let bsize = block.len();
    let row_count = u16::from_le_bytes([block[bsize - 2], block[bsize - 1]]) as usize;
    let payload = &block[..bsize - 2];
    let mut offset = 0usize;
    let mut rows = Vec::with_capacity(row_count);
    for _ in 0..row_count {
        let mut values = Vec::with_capacity(schema.len());
        for dtype in schema {
            values.push(decode_scratch_value(dtype, payload, &mut offset)?);
        }
        rows.push(Row { values });
    }
    Ok(rows)
}

/// Write a sorted run to scratch, return its location metadata.
fn write_run_to_scratch(
    rows: &[Row],
    schema: &[DataType],
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
) -> Result<ScratchRunMeta> {
    let block_size = disk_client.borrow().block_size;
    let data = encode_rows_to_scratch_blocks(rows, schema, block_size);
    let num_blocks = (data.len() / block_size) as u64;
    let start_block = scratch.alloc(num_blocks);
    disk_client.borrow_mut().put_blocks(start_block, &data)?;
    Ok(ScratchRunMeta { start_block, num_blocks })
}

/// Metadata for one sorted run stored in scratch space.
struct ScratchRunMeta {
    start_block: u64,
    num_blocks: u64,
}

/// Per-run cursor state for k-way merge.
struct RunState {
    block_offset: u64,
    rows_buffer: Vec<Row>,
    row_idx: usize,
}

/// Heap entry for k-way merge; wrapped in Reverse so BinaryHeap acts as min-heap.
struct HeapEntry {
    key_values: Vec<Data>,
    ascending: Vec<bool>,
    run_idx: usize,
    row: Row,
}

impl HeapEntry {
    fn cmp_key(&self, other: &Self) -> Ordering {
        for i in 0..self.key_values.len() {
            let ord = self.key_values[i]
                .partial_cmp(&other.key_values[i])
                .unwrap_or(Ordering::Equal);
            let ord = if self.ascending[i] { ord } else { ord.reverse() };
            if ord != Ordering::Equal { return ord; }
        }
        self.run_idx.cmp(&other.run_idx)
    }
}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool { self.cmp_key(other) == Ordering::Equal }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering { self.cmp_key(other) }
}

/// A RowSource that k-way merges multiple sorted scratch runs.
struct MergeSortSource {
    columns: Vec<ColumnMeta>,
    schema: Vec<DataType>,
    sort_col_indices: Vec<usize>,
    ascending: Vec<bool>,
    runs: Vec<ScratchRunMeta>,
    run_states: Vec<RunState>,
    disk_client: SharedDiskClient,
    block_size: usize,
    heap: BinaryHeap<std::cmp::Reverse<HeapEntry>>,
}

impl MergeSortSource {
    fn new(
        columns: Vec<ColumnMeta>,
        schema: Vec<DataType>,
        sort_col_indices: Vec<usize>,
        ascending: Vec<bool>,
        runs: Vec<ScratchRunMeta>,
        disk_client: SharedDiskClient,
        block_size: usize,
    ) -> Result<Self> {
        let mut run_states: Vec<RunState> = runs.iter().map(|_| RunState {
            block_offset: 0,
            rows_buffer: Vec::new(),
            row_idx: 0,
        }).collect();

        let mut heap = BinaryHeap::new();
        for run_idx in 0..runs.len() {
            if let Some(row) = Self::pull_next(
                run_idx, &runs, &mut run_states[run_idx], &schema, &disk_client, block_size,
            )? {
                let key_values: Vec<Data> = sort_col_indices.iter()
                    .map(|&i| row.values[i].clone()).collect();
                heap.push(std::cmp::Reverse(HeapEntry {
                    key_values, ascending: ascending.clone(), run_idx, row,
                }));
            }
        }

        Ok(Self { columns, schema, sort_col_indices, ascending, runs, run_states,
                  disk_client, block_size, heap })
    }

    fn pull_next(
        run_idx: usize,
        runs: &[ScratchRunMeta],
        state: &mut RunState,
        schema: &[DataType],
        disk_client: &SharedDiskClient,
        _block_size: usize,
    ) -> Result<Option<Row>> {
        if state.row_idx < state.rows_buffer.len() {
            let row = state.rows_buffer[state.row_idx].clone();
            state.row_idx += 1;
            return Ok(Some(row));
        }
        let run = &runs[run_idx];
        if state.block_offset >= run.num_blocks { return Ok(None); }
        let block_data = disk_client.borrow_mut()
            .get_blocks(run.start_block + state.block_offset, 1)?;
        state.block_offset += 1;
        state.rows_buffer = decode_scratch_block(&block_data, schema)?;
        state.row_idx = 0;
        if state.rows_buffer.is_empty() { return Ok(None); }
        let row = state.rows_buffer[state.row_idx].clone();
        state.row_idx += 1;
        Ok(Some(row))
    }
}

impl RowSource for MergeSortSource {
    fn columns(&self) -> &[ColumnMeta] { &self.columns }

    fn next_row(&mut self) -> Result<Option<Row>> {
        let Some(std::cmp::Reverse(entry)) = self.heap.pop() else {
            return Ok(None);
        };
        let run_idx = entry.run_idx;
        if let Some(next_row) = Self::pull_next(
            run_idx, &self.runs, &mut self.run_states[run_idx],
            &self.schema, &self.disk_client, self.block_size,
        )? {
            let key_values: Vec<Data> = self.sort_col_indices.iter()
                .map(|&i| next_row.values[i].clone()).collect();
            self.heap.push(std::cmp::Reverse(HeapEntry {
                key_values, ascending: self.ascending.clone(), run_idx, row: next_row,
            }));
        }
        Ok(Some(entry.row))
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

// ── External Merge Sort ───────────────────────────────────────────────────────

/// Sort `source` using disk scratch for runs that exceed memory_limit_mb / 2.
fn external_sort(
    sort: &SortData,
    source: Box<dyn RowSource>,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<Box<dyn RowSource>> {
    let columns = source.columns().to_vec();
    let schema: Vec<DataType> = columns.iter().map(|c| c.data_type.clone()).collect();

    // Compile sort specs to column indices once
    let col_idx = build_column_index(&columns);
    let sort_specs: Vec<(usize, bool)> = sort.sort_specs.iter().map(|s| {
        let i = *col_idx.get(s.column_name.as_str())
            .ok_or_else(|| anyhow!("Unknown sort column {}", s.column_name))?;
        Ok((i, s.ascending))
    }).collect::<Result<_>>()?;
    let sort_col_indices: Vec<usize> = sort_specs.iter().map(|(i, _)| *i).collect();
    let ascending: Vec<bool> = sort_specs.iter().map(|(_, a)| *a).collect();

    // Memory budget: use half the limit for in-memory runs
    let row_est = scratch_row_size_estimate(&schema);
    let budget_bytes = ((memory_limit_mb * 1024 * 1024) / 2) as usize;
    let run_capacity = (budget_bytes / row_est).max(1024);

    let block_size = disk_client.borrow().block_size;
    let mut runs: Vec<ScratchRunMeta> = Vec::new();
    let mut current_run: Vec<Row> = Vec::with_capacity(run_capacity);
    let mut source = source;

    loop {
        match source.next_row()? {
            None => break,
            Some(row) => {
                current_run.push(row);
                if current_run.len() >= run_capacity {
                    sort_run(&mut current_run, &sort_specs);
                    let meta = write_run_to_scratch(
                        &current_run, &schema, disk_client.clone(), scratch,
                    )?;
                    runs.push(meta);
                    current_run.clear();
                }
            }
        }
    }

    // Sort the last (possibly only) run
    if !current_run.is_empty() {
        sort_run(&mut current_run, &sort_specs);
        if runs.is_empty() {
            // Fast path: everything fit in memory — no scratch I/O needed
            return Ok(Box::new(MaterializedSource::new(ResultSet { columns, rows: current_run })));
        }
        let meta = write_run_to_scratch(&current_run, &schema, disk_client.clone(), scratch)?;
        runs.push(meta);
    }

    if runs.is_empty() {
        return Ok(Box::new(MaterializedSource::new(ResultSet { columns, rows: vec![] })));
    }

    Ok(Box::new(MergeSortSource::new(
        columns, schema, sort_col_indices, ascending, runs, disk_client, block_size,
    )?))
}

fn sort_run(rows: &mut Vec<Row>, sort_specs: &[(usize, bool)]) {
    rows.sort_by(|a, b| {
        for (idx, ascending) in sort_specs {
            let ord = a.values[*idx].partial_cmp(&b.values[*idx]).unwrap_or(Ordering::Equal);
            let ord = if *ascending { ord } else { ord.reverse() };
            if ord != Ordering::Equal { return ord; }
        }
        Ordering::Equal
    });
}

// ── Hash Join ────────────────────────────────────────────────────────────────

/// Collect all non-Cross leaf QueryOps from an arbitrarily deep left-linear cross chain.
fn collect_cross_leaves<'a>(op: &'a QueryOp, out: &mut Vec<&'a QueryOp>) {
    if let QueryOp::Cross(cross) = op {
        collect_cross_leaves(&cross.left, out);
        collect_cross_leaves(&cross.right, out);
    } else {
        out.push(op);
    }
}

/// Execute Filter(Cross(...)) using a greedy N-way hash join with predicate pushdown.
///
/// Algorithm:
///   1. Flatten the cross chain into N leaf ops.
///   2. Execute + materialize each leaf.
///   3. Push single-table predicates down: apply them immediately on each leaf's rows.
///   4. Greedy hash join: while any leaf has an equi-join predicate with the accumulated
///      result, hash-join it in (build on leaf, probe on accumulated).
///   5. Any leaf still un-joined gets cross-producted in (expected to be tiny tables).
///   6. Apply ALL filter predicates on the final result (idempotent, catches any missed).
fn execute_filter_cross(
    filter: &FilterData,
    cross: &CrossData,
    ctx: &DbContext,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<Box<dyn RowSource>> {
    // 1. Flatten cross chain
    let mut leaf_ops: Vec<&QueryOp> = Vec::new();
    collect_cross_leaves(&cross.left, &mut leaf_ops);
    collect_cross_leaves(&cross.right, &mut leaf_ops);
    let n = leaf_ops.len();

    // 2. Execute + materialize each leaf
    let mut leaf_data: Vec<Option<(Vec<ColumnMeta>, Vec<Row>)>> = leaf_ops
        .iter()
        .map(|op| {
            let rs = materialize_source(execute_query_op(
                op, ctx, disk_client.clone(), scratch, memory_limit_mb,
            )?)?;
            Ok(Some((rs.columns, rs.rows)))
        })
        .collect::<Result<_>>()?;

    // 3. Build col_name → leaf_index map
    let mut col_to_leaf: HashMap<String, usize> = HashMap::new();
    for (li, opt) in leaf_data.iter().enumerate() {
        if let Some((cols, _)) = opt {
            for c in cols {
                col_to_leaf.insert(c.name.clone(), li);
            }
        }
    }

    // 4. Predicate pushdown: apply single-table predicates on each leaf
    for li in 0..n {
        let col_idx = {
            let (cols, _) = leaf_data[li].as_ref().unwrap();
            build_column_index(cols)
        };
        let (_, rows) = leaf_data[li].as_mut().unwrap();
        rows.retain(|row| {
            filter.predicates.iter().all(|pred| {
                let left_leaf = col_to_leaf.get(&pred.column_name).copied();
                let right_leaf = match &pred.value {
                    ComparisionValue::Column(c) => col_to_leaf.get(c.as_str()).copied(),
                    _ => None,
                };
                match (left_leaf, right_leaf) {
                    (Some(ll), None) if ll == li =>
                        predicate_matches(pred, row, &col_idx).unwrap_or(false),
                    (Some(ll), Some(rl)) if ll == li && rl == li =>
                        predicate_matches(pred, row, &col_idx).unwrap_or(false),
                    _ => true,
                }
            })
        });
    }

    // 5. Greedy hash join
    let (init_cols, init_rows) = leaf_data[0].take().unwrap();
    let mut acc_cols: Vec<ColumnMeta> = init_cols;
    let mut acc_rows: Vec<Row> = init_rows;
    let mut joined = vec![false; n];
    joined[0] = true;

    let mut progress = true;
    while progress {
        progress = false;
        'outer: for li in 0..n {
            if joined[li] { continue; }
            let (leaf_cols, leaf_rows) = leaf_data[li].as_ref().unwrap();

            let acc_map: HashMap<&str, usize> = acc_cols.iter().enumerate()
                .map(|(i, c)| (c.name.as_str(), i)).collect();
            let leaf_map: HashMap<&str, usize> = leaf_cols.iter().enumerate()
                .map(|(i, c)| (c.name.as_str(), i)).collect();

            // Find first equi-join predicate between accumulated and leaf li
            let mut join_keys: Option<(usize, usize)> = None;
            for pred in &filter.predicates {
                if !matches!(pred.operator, ComparisionOperator::EQ) { continue; }
                if let ComparisionValue::Column(other) = &pred.value {
                    if let (Some(&ai), Some(&li_idx)) = (
                        acc_map.get(pred.column_name.as_str()),
                        leaf_map.get(other.as_str()),
                    ) {
                        join_keys = Some((ai, li_idx));
                        break;
                    }
                    if let (Some(&li_idx), Some(&ai)) = (
                        leaf_map.get(pred.column_name.as_str()),
                        acc_map.get(other.as_str()),
                    ) {
                        join_keys = Some((ai, li_idx));
                        break;
                    }
                }
            }

            if let Some((acc_key, leaf_key)) = join_keys {
                // Build hash table on leaf (smaller/next) side
                let mut ht: HashMap<DataKey, Vec<Row>> = HashMap::new();
                for row in leaf_rows {
                    ht.entry(DataKey(row.values[leaf_key].clone()))
                        .or_default()
                        .push(row.clone());
                }
                // Probe with accumulated rows
                let mut new_rows: Vec<Row> = Vec::new();
                for acc_row in &acc_rows {
                    let key = DataKey(acc_row.values[acc_key].clone());
                    if let Some(matches) = ht.get(&key) {
                        for leaf_row in matches {
                            let mut values = acc_row.values.clone();
                            values.extend_from_slice(&leaf_row.values);
                            new_rows.push(Row { values });
                        }
                    }
                }
                let mut new_cols = acc_cols.clone();
                new_cols.extend_from_slice(leaf_cols);
                acc_rows = new_rows;
                acc_cols = new_cols;
                leaf_data[li] = None;
                joined[li] = true;
                progress = true;
                break 'outer; // acc_cols changed — restart
            }
        }
    }

    // 6. Cross-product any unjoined leaves (no eq-join pred — should be tiny tables)
    for li in 0..n {
        if joined[li] { continue; }
        let (leaf_cols, leaf_rows) = leaf_data[li].take().unwrap();
        let mut new_rows =
            Vec::with_capacity(acc_rows.len().saturating_mul(leaf_rows.len()));
        for acc_row in &acc_rows {
            for leaf_row in &leaf_rows {
                let mut values = acc_row.values.clone();
                values.extend_from_slice(&leaf_row.values);
                new_rows.push(Row { values });
            }
        }
        let mut new_cols = acc_cols.clone();
        new_cols.extend(leaf_cols);
        acc_rows = new_rows;
        acc_cols = new_cols;
    }

    // 7. Apply ALL filter predicates on the final result (catches cross-table / non-equi)
    let final_col_idx = build_column_index(&acc_cols);
    acc_rows.retain(|row| {
        filter.predicates.iter().all(|pred| {
            predicate_matches(pred, row, &final_col_idx).unwrap_or(false)
        })
    });

    Ok(Box::new(MaterializedSource::new(ResultSet {
        columns: acc_cols,
        rows: acc_rows,
    })))
}

fn execute_query_op(
    query_op: &QueryOp,
    ctx: &DbContext,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,                 //APal
    memory_limit_mb: u64,
) -> Result<Box<dyn RowSource>> {
    match query_op {
        QueryOp::Scan(ScanData { table_id }) => {
            let table = find_table(ctx, table_id)?;
            Ok(Box::new(ScanSource::new(table, disk_client, None)?))
        }
        QueryOp::Filter(filter) => {
            // Filter over a Cross chain → use N-way hash join with predicate pushdown
            if let QueryOp::Cross(cross) = filter.underlying.as_ref() {
                return execute_filter_cross(
                    filter, cross, ctx, disk_client, scratch, memory_limit_mb,
                );
            }
            let child = execute_query_op(&filter.underlying, ctx, disk_client, scratch, memory_limit_mb)?;
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

            let child = execute_query_op(&project.underlying, ctx, disk_client, scratch, memory_limit_mb)?;      //APal
            Ok(Box::new(ProjectSource::new(project, child)?))
        }
        QueryOp::Sort(sort) => {
            let child = execute_query_op(&sort.underlying, ctx, disk_client.clone(), scratch, memory_limit_mb)?;         //APal
            external_sort(sort, child, disk_client, scratch, memory_limit_mb)
        }
        QueryOp::Cross(cross) => {
            let left = materialize_source(execute_query_op(&cross.left, ctx, disk_client.clone(), scratch, memory_limit_mb)?)?;      //APal
            let right = materialize_source(execute_query_op(&cross.right, ctx, disk_client, scratch, memory_limit_mb)?)?;            //APal
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

    let anon_start = disk_client.borrow_mut().get_anon_start_block()?;  //APal
    
    let block_size = disk_client.borrow().block_size;                   //APal
    let mut scratch = ScratchAllocator::new(anon_start, block_size);    //APal

    let query = read_query(&mut monitor_reader)?;
    let memory_limit_mb = request_memory_limit(&mut monitor_reader, &mut monitor_writer)?;

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

    let result_source = execute_query_op(&query.root, &ctx, disk_client, &mut scratch, memory_limit_mb)?;        //APal
    write_result_to_monitor(result_source, &mut monitor_writer)?;

    Ok(())
}

fn main() -> Result<()> {
    db_main().with_context(|| "From Database")
}
