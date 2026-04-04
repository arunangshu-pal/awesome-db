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
    cmp::Ordering,
    collections::HashMap,
    io::{BufRead, BufReader, Write},
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

#[derive(Clone)]
struct ResultSet {
    columns: Vec<ColumnMeta>,
    rows: Vec<Row>,
}

impl ResultSet {
    fn column_index_map(&self) -> HashMap<&str, usize> {
        self.columns
            .iter()
            .enumerate()
            .map(|(idx, column)| (column.name.as_str(), idx))
            .collect()
    }
}

struct DiskClient<R: BufRead, W: Write> {
    reader: R,
    writer: W,
    block_size: usize,
}

impl<R: BufRead, W: Write> DiskClient<R, W> {
    fn new(mut reader: R, mut writer: W) -> Result<Self> {
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

fn decode_table_blocks(table: &TableSpec, bytes: &[u8], block_size: usize) -> Result<ResultSet> {
    let columns = table
        .column_specs
        .iter()
        .map(|column| ColumnMeta {
            name: column.column_name.clone(),
            data_type: column.data_type.clone(),
        })
        .collect::<Vec<_>>();

    let mut rows = Vec::new();

    for block in bytes.chunks_exact(block_size) {
        let row_count_offset = block_size - 2;
        let row_count = u16::from_le_bytes([block[row_count_offset], block[row_count_offset + 1]]);
        let mut offset = 0usize;
        let payload = &block[..row_count_offset];

        for _ in 0..row_count {
            let mut values = Vec::with_capacity(table.column_specs.len());
            for column in &table.column_specs {
                values.push(decode_value(&column.data_type, payload, &mut offset)?);
            }
            rows.push(Row { values });
        }
    }

    Ok(ResultSet { columns, rows })
}

fn comparison_value_to_data(value: &ComparisionValue, row: &Row, column_index: &HashMap<&str, usize>) -> Result<Data> {
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
    column_index: &HashMap<&str, usize>,
) -> Result<bool> {
    let left_idx = *column_index
        .get(predicate.column_name.as_str())
        .ok_or_else(|| anyhow!("Unknown filter column {}", predicate.column_name))?;
    let left = &row.values[left_idx];
    let right = comparison_value_to_data(&predicate.value, row, column_index)?;
    Ok(compare_data(left, &predicate.operator, &right))
}

fn execute_filter(filter: &FilterData, input: ResultSet) -> Result<ResultSet> {
    let ResultSet { columns, rows: input_rows } = input;
    let column_index = columns
        .iter()
        .enumerate()
        .map(|(idx, column)| (column.name.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut rows = Vec::new();

    for row in input_rows {
        let mut keep_row = true;
        for predicate in &filter.predicates {
            if !predicate_matches(predicate, &row, &column_index)? {
                keep_row = false;
                break;
            }
        }

        if keep_row {
            rows.push(row);
        }
    }

    Ok(ResultSet {
        columns,
        rows,
    })
}

fn execute_project(project: &ProjectData, input: ResultSet) -> Result<ResultSet> {
    let column_index = input.column_index_map();

    let projection = project
        .column_name_map
        .iter()
        .map(|(from, to)| {
            let idx = *column_index
                .get(from.as_str())
                .ok_or_else(|| anyhow!("Unknown project column {from}"))?;
            Ok((idx, to.clone(), input.columns[idx].data_type.clone()))
        })
        .collect::<Result<Vec<_>>>()?;

    let columns = projection
        .iter()
        .map(|(_, name, data_type)| ColumnMeta {
            name: name.clone(),
            data_type: data_type.clone(),
        })
        .collect();

    let rows = input
        .rows
        .into_iter()
        .map(|row| Row {
            values: projection
                .iter()
                .map(|(idx, _, _)| row.values[*idx].clone())
                .collect(),
        })
        .collect();

    Ok(ResultSet { columns, rows })
}

fn sort_rows(sort: &SortData, input: &mut ResultSet) -> Result<()> {
    let column_index = input.column_index_map();
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

    input.rows.sort_by(|left, right| {
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
    let ResultSet {
        columns: mut left_columns,
        rows: left_rows,
    } = left;
    let ResultSet {
        columns: right_columns,
        rows: right_rows,
    } = right;

    left_columns.extend(right_columns);

    let mut rows = Vec::with_capacity(left_rows.len().saturating_mul(right_rows.len()));

    for left_row in left_rows {
        for right_row in &right_rows {
            let mut values = left_row.values.clone();
            values.extend(right_row.values.clone());
            rows.push(Row { values });
        }
    }

    ResultSet {
        columns: left_columns,
        rows,
    }
}

fn execute_query_op<R: BufRead, W: Write>(
    query_op: &QueryOp,
    ctx: &DbContext,
    disk_client: &mut DiskClient<R, W>,
) -> Result<ResultSet> {
    match query_op {
        QueryOp::Scan(ScanData { table_id }) => {
            let table = find_table(ctx, table_id)?;
            let start_block = disk_client.get_file_start_block(&table.file_id)?;
            let num_blocks = disk_client.get_file_num_blocks(&table.file_id)?;
            let bytes = disk_client.get_blocks(start_block, num_blocks)?;
            decode_table_blocks(table, &bytes, disk_client.block_size)
        }
        QueryOp::Filter(filter) => {
            let input = execute_query_op(&filter.underlying, ctx, disk_client)?;
            execute_filter(filter, input)
        }
        QueryOp::Project(project) => {
            let input = execute_query_op(&project.underlying, ctx, disk_client)?;
            execute_project(project, input)
        }
        QueryOp::Sort(sort) => {
            let mut input = execute_query_op(&sort.underlying, ctx, disk_client)?;
            sort_rows(sort, &mut input)?;
            Ok(input)
        }
        QueryOp::Cross(cross) => {
            let left = execute_query_op(&cross.left, ctx, disk_client)?;
            let right = execute_query_op(&cross.right, ctx, disk_client)?;
            Ok(execute_cross(left, right))
        }
    }
}

fn data_to_output(data: &Data) -> String {
    match data {
        Data::Int32(value) => value.to_string(),
        Data::Int64(value) => value.to_string(),
        Data::Float32(value) => value.to_string(),
        Data::Float64(value) => value.to_string(),
        Data::String(value) => value.clone(),
    }
}

fn write_result_to_monitor(result: &ResultSet, monitor_out: &mut impl Write) -> Result<()> {
    monitor_out.write_all(b"validate\n")?;

    for row in &result.rows {
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
    let (monitor_in, mut monitor_out) = setup_monitor_io();

    let disk_reader = BufReader::new(disk_in);
    let mut disk_client = DiskClient::new(disk_reader, disk_out)?;

    let mut monitor_reader = BufReader::new(monitor_in);
    let query = read_query(&mut monitor_reader)?;
    let _memory_limit_mb = request_memory_limit(&mut monitor_reader, &mut monitor_out)?;

    let result = execute_query_op(&query.root, &ctx, &mut disk_client)?;
    write_result_to_monitor(&result, &mut monitor_out)?;

    Ok(())
}

fn main() -> Result<()> {
    db_main().with_context(|| "From Database")
}
