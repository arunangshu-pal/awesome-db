use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use common::{
    Data, DataType,
    query::{
        ComparisionOperator, ComparisionValue, CrossData, FilterData, Predicate, ProjectData, Query,
        QueryOp, ScanData, SortData,
    },
};
use db_config::{DbContext, statistics::ColumnStat, table::TableSpec};
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
const SCRATCH_PREFETCH_BLOCKS: u64 = 32;
const SCRATCH_WRITE_BUF: usize = 2048;

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
#[derive(Clone)]
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
        block_size: usize,
    ) -> Result<Option<Row>> {
        if state.row_idx < state.rows_buffer.len() {
            let row = state.rows_buffer[state.row_idx].clone();
            state.row_idx += 1;
            return Ok(Some(row));
        }
        let run = &runs[run_idx];
        if state.block_offset >= run.num_blocks { return Ok(None); }
        // Batch-prefetch multiple blocks at once to reduce rotational latency
        let remaining = run.num_blocks - state.block_offset;
        let fetch_count = remaining.min(SCRATCH_PREFETCH_BLOCKS);
        let batch_data = disk_client.borrow_mut()
            .get_blocks(run.start_block + state.block_offset, fetch_count)?;
        state.block_offset += fetch_count;
        // Decode all fetched blocks into the buffer
        let mut all_rows = Vec::new();
        for block in batch_data.chunks_exact(block_size) {
            all_rows.extend(decode_scratch_block(block, schema)?);
        }
        state.rows_buffer = all_rows;
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

struct ScratchRunsSource {
    columns: Vec<ColumnMeta>,
    schema: Vec<DataType>,
    runs: Vec<ScratchRunMeta>,
    current_run_idx: usize,
    current_block_offset: u64,
    current_rows: std::vec::IntoIter<Row>,
    disk_client: SharedDiskClient,
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

impl ScratchRunsSource {
    fn new(
        columns: Vec<ColumnMeta>,
        schema: Vec<DataType>,
        runs: Vec<ScratchRunMeta>,
        disk_client: SharedDiskClient,
    ) -> Self {
        Self {
            columns,
            schema,
            runs,
            current_run_idx: 0,
            current_block_offset: 0,
            current_rows: Vec::new().into_iter(),
            disk_client,
        }
    }
}

impl RowSource for ScratchRunsSource {
    fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Row>> {
        loop {
            if let Some(row) = self.current_rows.next() {
                return Ok(Some(row));
            }

            if self.current_run_idx >= self.runs.len() {
                return Ok(None);
            }

            let run = &self.runs[self.current_run_idx];
            if self.current_block_offset >= run.num_blocks {
                self.current_run_idx += 1;
                self.current_block_offset = 0;
                continue;
            }

            // Batch-prefetch multiple blocks to reduce I/O operations
            let remaining = run.num_blocks - self.current_block_offset;
            let fetch_count = remaining.min(SCRATCH_PREFETCH_BLOCKS);
            let batch = self
                .disk_client
                .borrow_mut()
                .get_blocks(run.start_block + self.current_block_offset, fetch_count)?;
            self.current_block_offset += fetch_count;
            let block_size = self.disk_client.borrow().block_size;
            let mut all_rows = Vec::new();
            for block in batch.chunks_exact(block_size) {
                all_rows.extend(decode_scratch_block(block, &self.schema)?);
            }
            self.current_rows = all_rows.into_iter();
        }
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

fn collect_required_columns_for_project(
    project: &ProjectData,
    required_columns: &mut HashSet<String>,
) {
    for (from, _) in &project.column_name_map {
        required_columns.insert(from.clone());
    }
}

fn collect_required_columns_for_sort(
    sort: &SortData,
    required_columns: &mut HashSet<String>,
) {
    for spec in &sort.sort_specs {
        required_columns.insert(spec.column_name.clone());
    }
}

fn inferred_columns_for_query_op(query_op: &QueryOp, ctx: &DbContext) -> Result<Vec<ColumnMeta>> {
    match query_op {
        QueryOp::Scan(ScanData { table_id }) => {
            let table = find_table(ctx, table_id)?;
            Ok(table
                .column_specs
                .iter()
                .map(|column| ColumnMeta {
                    name: column.column_name.clone(),
                    data_type: column.data_type.clone(),
                })
                .collect())
        }
        QueryOp::Filter(filter) => inferred_columns_for_query_op(&filter.underlying, ctx),
        QueryOp::Sort(sort) => inferred_columns_for_query_op(&sort.underlying, ctx),
        QueryOp::Project(project) => {
            let input_columns = inferred_columns_for_query_op(&project.underlying, ctx)?;
            let input_index = build_column_index(&input_columns);
            project
                .column_name_map
                .iter()
                .map(|(from, to)| {
                    let idx = *input_index
                        .get(from.as_str())
                        .ok_or_else(|| anyhow!("Unknown project column {from}"))?;
                    Ok(ColumnMeta {
                        name: to.clone(),
                        data_type: input_columns[idx].data_type.clone(),
                    })
                })
                .collect()
        }
        QueryOp::Cross(cross) => {
            let mut columns = inferred_columns_for_query_op(&cross.left, ctx)?;
            columns.extend(inferred_columns_for_query_op(&cross.right, ctx)?);
            Ok(columns)
        }
    }
}

fn estimated_output_cardinality(
    query_op: &QueryOp,
    output_column_name: &str,
    ctx: &DbContext,
) -> Option<usize> {
    match query_op {
        QueryOp::Scan(ScanData { table_id }) => {
            let table = find_table(ctx, table_id).ok()?;
            let column = table
                .column_specs
                .iter()
                .find(|column| column.column_name == output_column_name)?;
            column.stats.as_ref()?.iter().find_map(|stat| match stat {
                ColumnStat::CardinalityStat(value) => Some(value.0 as usize),
                _ => None,
            })
        }
        QueryOp::Filter(filter) => estimated_output_cardinality(&filter.underlying, output_column_name, ctx),
        QueryOp::Sort(sort) => estimated_output_cardinality(&sort.underlying, output_column_name, ctx),
        QueryOp::Project(project) => {
            let source_name = project
                .column_name_map
                .iter()
                .find_map(|(from, to)| (to == output_column_name).then_some(from.as_str()))?;
            estimated_output_cardinality(&project.underlying, source_name, ctx)
        }
        QueryOp::Cross(cross) => estimated_output_cardinality(&cross.left, output_column_name, ctx)
            .or_else(|| estimated_output_cardinality(&cross.right, output_column_name, ctx)),
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

fn predicate_is_ready(predicate: &Predicate, column_index: &HashMap<String, usize>) -> bool {
    if !column_index.contains_key(predicate.column_name.as_str()) {
        return false;
    }
    match &predicate.value {
        ComparisionValue::Column(name) => column_index.contains_key(name.as_str()),
        _ => true,
    }
}

fn project_rows_in_place(
    columns: &mut Vec<ColumnMeta>,
    rows: &mut Vec<Row>,
    keep_names: &HashSet<String>,
) {
    let keep_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter_map(|(idx, column)| keep_names.contains(column.name.as_str()).then_some(idx))
        .collect();

    if keep_indices.len() == columns.len() {
        return;
    }

    let new_columns: Vec<ColumnMeta> = keep_indices
        .iter()
        .map(|idx| columns[*idx].clone())
        .collect();

    for row in rows.iter_mut() {
        row.values = keep_indices
            .iter()
            .map(|idx| row.values[*idx].clone())
            .collect();
    }

    *columns = new_columns;
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

fn out_of_core_cross(
    left_src: Box<dyn RowSource>,
    right_src: Box<dyn RowSource>,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<Box<dyn RowSource>> {
    let left_leaf = materialize_filtered_source_to_scratch(left_src, &[], disk_client.clone(), scratch, memory_limit_mb)?;
    let right_leaf = materialize_filtered_source_to_scratch(right_src, &[], disk_client.clone(), scratch, memory_limit_mb)?;
    
    let mut output_columns = left_leaf.columns.clone();
    output_columns.extend(right_leaf.columns.clone());
    let output_schema: Vec<DataType> = output_columns.iter().map(|c| c.data_type.clone()).collect();
    
    let mut out_buf: Vec<Row> = Vec::with_capacity(SCRATCH_WRITE_BUF);
    let mut runs = Vec::new();
    
    let mut left_stream = make_leaf_source(&left_leaf, disk_client.clone());
    while let Some(left_row) = left_stream.next_row()? {
        let mut right_stream = make_leaf_source(&right_leaf, disk_client.clone());
        while let Some(right_row) = right_stream.next_row()? {
            let mut values = left_row.values.clone();
            values.extend_from_slice(&right_row.values);
            out_buf.push(Row { values });
            if out_buf.len() >= SCRATCH_WRITE_BUF {
                runs.push(write_run_to_scratch(&out_buf, &output_schema, disk_client.clone(), scratch)?);
                out_buf.clear();
            }
        }
    }
    if !out_buf.is_empty() {
        runs.push(write_run_to_scratch(&out_buf, &output_schema, disk_client.clone(), scratch)?);
    }
    
    Ok(Box::new(ScratchRunsSource::new(output_columns, output_schema, runs, disk_client)))
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

    // Memory budget: use 3/4 of the limit for in-memory runs (runs written to scratch then freed)
    let row_est = scratch_row_size_estimate(&schema);
    let budget_bytes = ((memory_limit_mb * 1024 * 1024) * 3 / 4) as usize;
    let run_capacity = (budget_bytes / row_est).clamp(256, 32768);

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

fn execute_join_leaf(
    query_op: &QueryOp,
    ctx: &DbContext,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
    required_columns: Option<&HashSet<String>>,
) -> Result<Box<dyn RowSource>> {
    match query_op {
        QueryOp::Scan(ScanData { table_id }) => {
            let table = find_table(ctx, table_id)?;
            Ok(Box::new(ScanSource::new(table, disk_client, required_columns)?))
        }
        QueryOp::Filter(filter) => {
            if let QueryOp::Scan(ScanData { table_id }) = filter.underlying.as_ref() {
                let table = find_table(ctx, table_id)?;
                let mut leaf_required = required_columns.cloned().unwrap_or_default();
                collect_required_columns_for_filter(filter, &mut leaf_required);
                let scan = ScanSource::new(table, disk_client, Some(&leaf_required))?;
                return Ok(Box::new(FilterSource::new(filter, Box::new(scan))));
            }
            execute_query_op(query_op, ctx, disk_client, scratch, memory_limit_mb)
        }
        _ => execute_query_op(query_op, ctx, disk_client, scratch, memory_limit_mb),
    }
}

/// Execute Filter(Cross(...)) using a streaming N-way grace hash join with predicate pushdown.
///
/// Algorithm:
///   1. Flatten the cross chain into N leaf ops.
///   2. Peek at each leaf to get its column schema (without materializing rows).
///   3. Build col_name → leaf_index map from schemas.
///   4. Apply predicate pushdown: compile single-table predicates per leaf.
///   5. Greedy join order: find equi-join pairs. For each pair:
///      a. Stream-partition BOTH sides directly to scratch (never hold >1 partition in memory).
///      b. Join partition-by-partition.
///      c. The result becomes the new "accumulated" side.
///   6. Any unjoined leaf (no join predicate) gets cross-producted — expected tiny tables.
///   7. Apply final filter on accumulated result.
fn execute_filter_cross(
    filter: &FilterData,
    cross: &CrossData,
    ctx: &DbContext,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
    required_columns: Option<&HashSet<String>>,
) -> Result<Box<dyn RowSource>> {
    // 1. Flatten cross chain
    let mut leaf_ops: Vec<&QueryOp> = Vec::new();
    collect_cross_leaves(&cross.left, &mut leaf_ops);
    collect_cross_leaves(&cross.right, &mut leaf_ops);
    let n = leaf_ops.len();

    let leaf_output_columns: Vec<Vec<ColumnMeta>> = leaf_ops
        .iter()
        .map(|op| inferred_columns_for_query_op(op, ctx))
        .collect::<Result<_>>()?;

    let mut col_to_leaf: HashMap<String, usize> = HashMap::new();
    for (li, cols) in leaf_output_columns.iter().enumerate() {
        for c in cols {
            col_to_leaf.insert(c.name.clone(), li);
        }
    }

    let mut per_leaf_required = vec![HashSet::new(); n];
    let mut join_required = HashSet::new();
    collect_required_columns_for_filter(filter, &mut join_required);
    if let Some(required) = required_columns {
        join_required.extend(required.iter().cloned());
    }
    for column_name in join_required {
        if let Some(&leaf_idx) = col_to_leaf.get(column_name.as_str()) {
            per_leaf_required[leaf_idx].insert(column_name);
        }
    }

    // 2. Build each leaf as a RowSource (streaming, not materialized yet).
    //    Also capture the column schema from each source.
    //    We need to peek at columns before consuming rows, so we create the sources
    //    and extract column metadata immediately.
    let mut leaf_sources: Vec<Option<Box<dyn RowSource>>> = leaf_ops
        .iter()
        .enumerate()
        .map(|(li, op)| {
            Ok(Some(execute_join_leaf(
                op,
                ctx,
                disk_client.clone(),
                scratch,
                memory_limit_mb,
                Some(&per_leaf_required[li]),
            )?))
        })
        .collect::<Result<_>>()?;

    let leaf_cols: Vec<Vec<ColumnMeta>> = leaf_sources.iter()
        .map(|s| s.as_ref().unwrap().columns().to_vec())
        .collect();

    // 3. Compile single-table filter predicates per leaf (for pushdown during streaming)
    let leaf_single_preds: Vec<Vec<CompiledPredicate>> = (0..n).map(|li| {
        let col_idx = build_column_index(&leaf_cols[li]);
        filter.predicates.iter().filter_map(|pred| {
            let left_leaf = col_to_leaf.get(&pred.column_name).copied();
            let right_leaf = match &pred.value {
                ComparisionValue::Column(c) => col_to_leaf.get(c.as_str()).copied(),
                _ => None,
            };
            let is_single = match (left_leaf, right_leaf) {
                (Some(ll), None) if ll == li => true,
                (Some(ll), Some(rl)) if ll == li && rl == li => true,
                _ => false,
            };
            if is_single {
                compile_predicate(pred, &col_idx).ok()
            } else {
                None
            }
        }).collect()
    }).collect();

    let leaf_materialized: Vec<MaterializedLeaf> = leaf_sources
        .into_iter()
        .enumerate()
        .map(|(li, source)| {
            materialize_filtered_source_to_scratch(
                source.unwrap(),
                &leaf_single_preds[li],
                disk_client.clone(),
                scratch,
                memory_limit_mb,
            )
        })
        .collect::<Result<_>>()?;

    let block_size = disk_client.borrow().block_size;
    let seed_idx = leaf_materialized
        .iter()
        .enumerate()
        .min_by_key(|(_, leaf)| leaf.row_count)
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let mut joined = vec![false; n];
    joined[seed_idx] = true;
    let mut acc_leaf = rewrite_leaf_to_scratch(
        &leaf_materialized[seed_idx],
        filter,
        &joined,
        &col_to_leaf,
        required_columns,
        disk_client.clone(),
        scratch,
        memory_limit_mb,
    )?;

    loop {
        let acc_map: HashMap<&str, usize> = acc_leaf
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.as_str(), i))
            .collect();
        let mut best_join: Option<(usize, usize, usize, usize)> = None;

        for li in 0..n {
            if joined[li] {
                continue;
            }

            let leaf_map: HashMap<&str, usize> = leaf_materialized[li]
                .columns
                .iter()
                .enumerate()
                .map(|(i, c)| (c.name.as_str(), i))
                .collect();

            let mut join_keys: Option<(usize, usize)> = None;
            for pred in &filter.predicates {
                if !matches!(pred.operator, ComparisionOperator::EQ) {
                    continue;
                }
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
                let leaf_key_name = &leaf_materialized[li].columns[leaf_key].name;
                let key_cardinality = estimated_output_cardinality(
                    leaf_ops[li],
                    leaf_key_name,
                    ctx,
                )
                .unwrap_or(1)
                .max(1);

                let better = best_join
                    .map(|(best_li, _, _, best_cardinality)| {
                        leaf_materialized[li].row_count * best_cardinality
                            < leaf_materialized[best_li].row_count * key_cardinality
                    })
                    .unwrap_or(true);
                if better {
                    best_join = Some((li, acc_key, leaf_key, key_cardinality));
                }
            }
        }

        let Some((li, acc_key, leaf_key, _)) = best_join else {
            break;
        };

        let mut output_columns = acc_leaf.columns.clone();
        output_columns.extend_from_slice(&leaf_materialized[li].columns);
        let joined_leaf = join_leaves_to_scratch(
            &acc_leaf,
            acc_key,
            &leaf_materialized[li],
            leaf_key,
            output_columns,
            disk_client.clone(),
            scratch,
            memory_limit_mb,
        )?;
        joined[li] = true;
        acc_leaf = rewrite_leaf_to_scratch(
            &joined_leaf,
            filter,
            &joined,
            &col_to_leaf,
            required_columns,
            disk_client.clone(),
            scratch,
            memory_limit_mb,
        )?;
    }

    // 5. Cross-product any unjoined leaves (no eq-join pred — expected tiny tables)
    for li in 0..n {
        if joined[li] { continue; }
        
        let mut new_cols = acc_leaf.columns.clone();
        new_cols.extend(leaf_materialized[li].columns.clone());
        let schema: Vec<DataType> = new_cols.iter().map(|c| c.data_type.clone()).collect();
        
        let mut runs = Vec::new();
        let mut out_buf: Vec<Row> = Vec::with_capacity(SCRATCH_WRITE_BUF);
        
        let mut acc_stream = make_leaf_source(&acc_leaf, disk_client.clone());
        while let Some(acc_row) = acc_stream.next_row()? {
            let mut leaf_stream = make_leaf_source(&leaf_materialized[li], disk_client.clone());
            while let Some(leaf_row) = leaf_stream.next_row()? {
                let mut values = acc_row.values.clone();
                values.extend_from_slice(&leaf_row.values);
                out_buf.push(Row { values });
                if out_buf.len() >= SCRATCH_WRITE_BUF {
                    runs.push(write_run_to_scratch(&out_buf, &schema, disk_client.clone(), scratch)?);
                    out_buf.clear();
                }
            }
        }
        if !out_buf.is_empty() {
            runs.push(write_run_to_scratch(&out_buf, &schema, disk_client.clone(), scratch)?);
        }

        let tmp_leaf = MaterializedLeaf {
            columns: new_cols.clone(),
            schema,
            runs,
            row_count: acc_leaf.row_count.saturating_mul(leaf_materialized[li].row_count),
        };
        joined[li] = true;
        acc_leaf = rewrite_leaf_to_scratch(
            &tmp_leaf,
            filter,
            &joined,
            &col_to_leaf,
            required_columns,
            disk_client.clone(),
            scratch,
            memory_limit_mb,
        )?;
    }

    // 6. Return accumulated result directly (rewrite already applied in the loop)
    //    All predicates that reference only joined leaves have already been applied
    //    by prior rewrite_leaf_to_scratch calls. Skip redundant final pass.
    Ok(Box::new(ScratchRunsSource::new(
        acc_leaf.columns,
        acc_leaf.schema,
        acc_leaf.runs,
        disk_client,
    )))
}

// ── Grace Hash Join ───────────────────────────────────────────────────────────

/// Hash key value to a partition number using a simple hash.
fn partition_of(val: &Data, num_partitions: usize) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    match val {
        Data::Int32(v)   => { 0u8.hash(&mut h); v.hash(&mut h); }
        Data::Int64(v)   => { 1u8.hash(&mut h); v.hash(&mut h); }
        Data::Float32(v) => { 2u8.hash(&mut h); v.to_bits().hash(&mut h); }
        Data::Float64(v) => { 3u8.hash(&mut h); v.to_bits().hash(&mut h); }
        Data::String(v)  => { 4u8.hash(&mut h); v.hash(&mut h); }
    }
    (h.finish() as usize) % num_partitions
}

/// Grace hash join between `left` (probe side) and `right` (build side).
///
/// If the build side fits in `budget_bytes`, fall back to a simple in-memory hash join.
/// Otherwise partition both sides to scratch, then join one partition at a time.
fn grace_hash_join(
    left: &[Row],   left_key:  usize,
    right: &[Row],  right_key: usize,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<Vec<Row>> {
    // Budget = 1/3 of memory limit for the build-side hash table of one partition.
    let budget_bytes = ((memory_limit_mb * 1024 * 1024) / 3) as usize;

    // Estimate right (build) side memory.
    // Simple heuristic: count bytes in encoded form (8 bytes per numeric, string len).
    let right_est: usize = right.iter().map(|r| {
        r.values.iter().map(|v| match v {
            Data::String(s) => s.len() + 2,
            _ => 8,
        }).sum::<usize>() + 24
    }).sum();

    if right_est <= budget_bytes {
        // Fast path: right side fits in memory — simple hash join.
        return Ok(simple_hash_join(left, left_key, right, right_key));
    }

    // Compute number of partitions so each partition's build side fits in budget.
    let num_partitions = ((right_est / budget_bytes) + 1).max(2).min(256);

    // Determine schemas from the rows (fall back to empty if no rows).
    let left_schema: Vec<DataType> = if let Some(r) = left.first() {
        r.values.iter().map(|v| match v {
            Data::Int32(_)   => DataType::Int32,
            Data::Int64(_)   => DataType::Int64,
            Data::Float32(_) => DataType::Float32,
            Data::Float64(_) => DataType::Float64,
            Data::String(_)  => DataType::String,
        }).collect()
    } else { vec![] };

    let right_schema: Vec<DataType> = if let Some(r) = right.first() {
        r.values.iter().map(|v| match v {
            Data::Int32(_)   => DataType::Int32,
            Data::Int64(_)   => DataType::Int64,
            Data::Float32(_) => DataType::Float32,
            Data::Float64(_) => DataType::Float64,
            Data::String(_)  => DataType::String,
        }).collect()
    } else { vec![] };

    let block_size = disk_client.borrow().block_size;

    // Partition left side into scratch.
    let mut left_part_bufs: Vec<Vec<Row>> = (0..num_partitions).map(|_| Vec::new()).collect();
    for row in left {
        let p = partition_of(&row.values[left_key], num_partitions);
        left_part_bufs[p].push(row.clone());
    }
    let left_runs: Vec<ScratchRunMeta> = left_part_bufs.iter().map(|rows| {
        write_run_to_scratch(rows, &left_schema, disk_client.clone(), scratch)
    }).collect::<Result<_>>()?;
    drop(left_part_bufs);

    // Partition right side into scratch.
    let mut right_part_bufs: Vec<Vec<Row>> = (0..num_partitions).map(|_| Vec::new()).collect();
    for row in right {
        let p = partition_of(&row.values[right_key], num_partitions);
        right_part_bufs[p].push(row.clone());
    }
    let right_runs: Vec<ScratchRunMeta> = right_part_bufs.iter().map(|rows| {
        write_run_to_scratch(rows, &right_schema, disk_client.clone(), scratch)
    }).collect::<Result<_>>()?;
    drop(right_part_bufs);

    // Join each partition pair.
    let mut output: Vec<Row> = Vec::new();
    for p in 0..num_partitions {
        // Load right partition (build side).
        let right_part = load_scratch_run(&right_runs[p], &right_schema, disk_client.clone(), block_size)?;
        if right_part.is_empty() { continue; }

        // Load left partition (probe side).
        let left_part = load_scratch_run(&left_runs[p], &left_schema, disk_client.clone(), block_size)?;
        if left_part.is_empty() { continue; }

        // In-memory hash join for this partition.
        let joined = simple_hash_join(&left_part, left_key, &right_part, right_key);
        output.extend(joined);
    }

    Ok(output)
}

/// Load all rows from a scratch run — batch-reads blocks for fewer I/O ops.
fn load_scratch_run(
    run: &ScratchRunMeta,
    schema: &[DataType],
    disk_client: SharedDiskClient,
    block_size: usize,
) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    let mut block_offset = 0u64;
    while block_offset < run.num_blocks {
        let remaining = run.num_blocks - block_offset;
        let fetch_count = remaining.min(SCRATCH_PREFETCH_BLOCKS);
        let batch_data = disk_client.borrow_mut()
            .get_blocks(run.start_block + block_offset, fetch_count)?;
        block_offset += fetch_count;
        for block in batch_data.chunks_exact(block_size) {
            rows.extend(decode_scratch_block(block, schema)?);
        }
    }
    Ok(rows)
}

fn load_scratch_runs(
    runs: &[ScratchRunMeta],
    schema: &[DataType],
    disk_client: SharedDiskClient,
    block_size: usize,
) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for run in runs {
        rows.extend(load_scratch_run(run, schema, disk_client.clone(), block_size)?);
    }
    Ok(rows)
}

#[derive(Clone)]
struct MaterializedLeaf {
    columns: Vec<ColumnMeta>,
    schema: Vec<DataType>,
    runs: Vec<ScratchRunMeta>,
    row_count: usize,
}

fn materialize_filtered_source_to_scratch(
    mut source: Box<dyn RowSource>,
    predicates: &[CompiledPredicate],
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<MaterializedLeaf> {
    let columns = source.columns().to_vec();
    let schema: Vec<DataType> = columns.iter().map(|column| column.data_type.clone()).collect();
    let row_capacity = ((((memory_limit_mb * 1024 * 1024) / 4) as usize)
        / scratch_row_size_estimate(&schema))
        .clamp(256, 32768);

    let mut runs = Vec::new();
    let mut row_count = 0usize;
    let mut buffer = Vec::with_capacity(row_capacity);

    while let Some(row) = source.next_row()? {
        if predicates
            .iter()
            .all(|predicate| compiled_predicate_matches(predicate, &row).unwrap_or(false))
        {
            row_count += 1;
            buffer.push(row);
            if buffer.len() >= row_capacity {
                runs.push(write_run_to_scratch(
                    &buffer,
                    &schema,
                    disk_client.clone(),
                    scratch,
                )?);
                buffer.clear();
            }
        }
    }

    if !buffer.is_empty() {
        runs.push(write_run_to_scratch(
            &buffer,
            &schema,
            disk_client,
            scratch,
        )?);
    }

    Ok(MaterializedLeaf {
        columns,
        schema,
        runs,
        row_count,
    })
}

fn make_leaf_source(leaf: &MaterializedLeaf, disk_client: SharedDiskClient) -> Box<dyn RowSource> {
    Box::new(ScratchRunsSource::new(
        leaf.columns.clone(),
        leaf.schema.clone(),
        leaf.runs.clone(),
        disk_client,
    ))
}

fn compute_keep_names(
    filter: &FilterData,
    joined: &[bool],
    col_to_leaf: &HashMap<String, usize>,
    required_columns: Option<&HashSet<String>>,
) -> HashSet<String> {
    let mut keep_names = required_columns.cloned().unwrap_or_default();

    for predicate in &filter.predicates {
        let left_leaf = col_to_leaf.get(predicate.column_name.as_str()).copied();
        let right_leaf = match &predicate.value {
            ComparisionValue::Column(name) => col_to_leaf.get(name.as_str()).copied(),
            _ => None,
        };

        if let (Some(joined_leaf), Some(unjoined_leaf)) = (left_leaf, right_leaf) {
            if joined[joined_leaf] && !joined[unjoined_leaf] {
                keep_names.insert(predicate.column_name.clone());
            }
        }

        if let (Some(unjoined_leaf), Some(joined_leaf)) = (left_leaf, right_leaf) {
            if !joined[unjoined_leaf] && joined[joined_leaf] {
                if let ComparisionValue::Column(name) = &predicate.value {
                    keep_names.insert(name.clone());
                }
            }
        }
    }

    keep_names
}

fn partition_source_to_scratch(
    mut source: Box<dyn RowSource>,
    key_idx: usize,
    schema: &[DataType],
    num_partitions: usize,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
) -> Result<Vec<Vec<ScratchRunMeta>>> {
    const WRITE_BUF: usize = SCRATCH_WRITE_BUF;

    let mut bufs: Vec<Vec<Row>> = (0..num_partitions).map(|_| Vec::new()).collect();
    let mut runs: Vec<Vec<ScratchRunMeta>> = (0..num_partitions).map(|_| Vec::new()).collect();

    while let Some(row) = source.next_row()? {
        let p = partition_of(&row.values[key_idx], num_partitions);
        bufs[p].push(row);
        if bufs[p].len() >= WRITE_BUF {
            let meta = write_run_to_scratch(&bufs[p], schema, disk_client.clone(), scratch)?;
            runs[p].push(meta);
            bufs[p].clear();
        }
    }

    for p in 0..num_partitions {
        if !bufs[p].is_empty() {
            let meta = write_run_to_scratch(&bufs[p], schema, disk_client.clone(), scratch)?;
            runs[p].push(meta);
            bufs[p].clear();
        }
    }

    Ok(runs)
}

fn join_leaves_to_scratch(
    left_leaf: &MaterializedLeaf,
    left_key: usize,
    right_leaf: &MaterializedLeaf,
    right_key: usize,
    output_columns: Vec<ColumnMeta>,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<MaterializedLeaf> {
    let budget_bytes = ((memory_limit_mb * 1024 * 1024) / 3) as usize;
    let right_est = right_leaf.row_count
        .saturating_mul(scratch_row_size_estimate(&right_leaf.schema));
    let num_partitions = if right_est <= budget_bytes {
        1usize
    } else {
        ((right_est / budget_bytes) + 1).max(2).min(256)
    };

    let left_parts = partition_source_to_scratch(
        make_leaf_source(left_leaf, disk_client.clone()),
        left_key,
        &left_leaf.schema,
        num_partitions,
        disk_client.clone(),
        scratch,
    )?;
    let right_parts = partition_source_to_scratch(
        make_leaf_source(right_leaf, disk_client.clone()),
        right_key,
        &right_leaf.schema,
        num_partitions,
        disk_client.clone(),
        scratch,
    )?;

    let output_schema: Vec<DataType> = output_columns
        .iter()
        .map(|column| column.data_type.clone())
        .collect();
    let block_size = disk_client.borrow().block_size;
    let mut runs = Vec::new();
    let mut row_count = 0usize;
    let mut out_buf: Vec<Row> = Vec::with_capacity(SCRATCH_WRITE_BUF);

    for p in 0..num_partitions {
        if right_parts[p].is_empty() || left_parts[p].is_empty() {
            continue;
        }
        
        let right_part_bytes: usize = right_parts[p].iter().map(|m| m.num_blocks as usize * block_size).sum();
        
        // Handle Skew: If Grace hash partition collision exceeds memory limits
        if right_part_bytes > budget_bytes {
            let mut left_src = ScratchRunsSource::new(
                left_leaf.columns.clone(),
                left_leaf.schema.clone(),
                left_parts[p].clone(),
                disk_client.clone(),
            );
            while let Some(left_row) = left_src.next_row()? {
                let mut right_src = ScratchRunsSource::new(
                    right_leaf.columns.clone(),
                    right_leaf.schema.clone(),
                    right_parts[p].clone(),
                    disk_client.clone(),
                );
                while let Some(right_row) = right_src.next_row()? {
                    if left_row.values[left_key] == right_row.values[right_key] {
                        let mut values = left_row.values.clone();
                        values.extend_from_slice(&right_row.values);
                        out_buf.push(Row { values });
                        row_count += 1;
                        if out_buf.len() >= SCRATCH_WRITE_BUF {
                            runs.push(write_run_to_scratch(&out_buf, &output_schema, disk_client.clone(), scratch)?);
                            out_buf.clear();
                        }
                    }
                }
            }
            continue;
        }

        let right_part =
            load_scratch_runs(&right_parts[p], &right_leaf.schema, disk_client.clone(), block_size)?;

        let mut ht: HashMap<DataKey, Vec<&Row>> = HashMap::new();
        for row in &right_part {
            ht.entry(DataKey(row.values[right_key].clone()))
                .or_default()
                .push(row);
        }

        let mut left_src = ScratchRunsSource::new(
            left_leaf.columns.clone(),
            left_leaf.schema.clone(),
            left_parts[p].clone(),
            disk_client.clone(),
        );

        while let Some(left_row) = left_src.next_row()? {
            let key = DataKey(left_row.values[left_key].clone());
            if let Some(matches) = ht.get(&key) {
                for right_row in matches {
                    let mut values = left_row.values.clone();
                    values.extend_from_slice(&right_row.values);
                    out_buf.push(Row { values });
                    row_count += 1;

                    if out_buf.len() >= SCRATCH_WRITE_BUF {
                        runs.push(write_run_to_scratch(
                            &out_buf,
                            &output_schema,
                            disk_client.clone(),
                            scratch,
                        )?);
                        out_buf.clear();
                    }
                }
            }
        }
    }

    if !out_buf.is_empty() {
        runs.push(write_run_to_scratch(
            &out_buf,
            &output_schema,
            disk_client,
            scratch,
        )?);
    }

    Ok(MaterializedLeaf {
        columns: output_columns,
        schema: output_schema,
        runs,
        row_count,
    })
}

fn rewrite_leaf_to_scratch(
    leaf: &MaterializedLeaf,
    filter: &FilterData,
    joined: &[bool],
    col_to_leaf: &HashMap<String, usize>,
    required_columns: Option<&HashSet<String>>,
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<MaterializedLeaf> {
    let keep_names = compute_keep_names(filter, joined, col_to_leaf, required_columns);
    let keep_indices: Vec<usize> = leaf
        .columns
        .iter()
        .enumerate()
        .filter_map(|(idx, column)| keep_names.contains(column.name.as_str()).then_some(idx))
        .collect();
    let columns: Vec<ColumnMeta> = keep_indices
        .iter()
        .map(|idx| leaf.columns[*idx].clone())
        .collect();
    let schema: Vec<DataType> = columns
        .iter()
        .map(|column| column.data_type.clone())
        .collect();
    let row_capacity = ((((memory_limit_mb * 1024 * 1024) / 4) as usize)
        / scratch_row_size_estimate(&schema))
        .clamp(256, 32768);
    let current_col_idx = build_column_index(&leaf.columns);

    let mut source = make_leaf_source(leaf, disk_client.clone());
    let mut runs = Vec::new();
    let mut row_count = 0usize;
    let mut buffer: Vec<Row> = Vec::with_capacity(row_capacity);

    while let Some(row) = source.next_row()? {
        if filter.predicates.iter().all(|predicate| {
            !predicate_is_ready(predicate, &current_col_idx)
                || predicate_matches(predicate, &row, &current_col_idx).unwrap_or(false)
        }) {
            row_count += 1;
            buffer.push(Row {
                values: keep_indices
                    .iter()
                    .map(|idx| row.values[*idx].clone())
                    .collect(),
            });

            if buffer.len() >= row_capacity {
                runs.push(write_run_to_scratch(
                    &buffer,
                    &schema,
                    disk_client.clone(),
                    scratch,
                )?);
                buffer.clear();
            }
        }
    }

    if !buffer.is_empty() {
        runs.push(write_run_to_scratch(
            &buffer,
            &schema,
            disk_client,
            scratch,
        )?);
    }

    Ok(MaterializedLeaf {
        columns,
        schema,
        runs,
        row_count,
    })
}

/// Simple in-memory hash join (build hash table on right, probe with left).
fn simple_hash_join(
    left: &[Row],  left_key:  usize,
    right: &[Row], right_key: usize,
) -> Vec<Row> {
    let mut ht: HashMap<DataKey, Vec<&Row>> = HashMap::new();
    for row in right {
        ht.entry(DataKey(row.values[right_key].clone()))
            .or_default()
            .push(row);
    }
    let mut output = Vec::new();
    for left_row in left {
        let key = DataKey(left_row.values[left_key].clone());
        if let Some(matches) = ht.get(&key) {
            for right_row in matches {
                let mut values = left_row.values.clone();
                values.extend_from_slice(&right_row.values);
                output.push(Row { values });
            }
        }
    }
    output
}

/// Grace hash join where the right side is a streaming RowSource (not yet materialized).
///
/// 1. Stream right source through predicate filter, writing each row into a scratch partition
///    determined by hash(row[right_key]).  Uses a per-partition write buffer (512 rows) to
///    avoid per-row disk writes.
/// 2. Partition the left (already materialized) side the same way.
/// 3. Join each partition pair in-memory.
fn streaming_grace_hash_join(
    left: &[Row],         left_key:    usize,
    left_schema:  &[DataType],
    mut right_src: Box<dyn RowSource>,
    right_key:   usize,
    right_schema: &[DataType],
    right_preds: &[CompiledPredicate],
    disk_client: SharedDiskClient,
    scratch: &mut ScratchAllocator,
    memory_limit_mb: u64,
) -> Result<Vec<Row>> {
    // Budget = 1/3 of memory limit for a single partition's hash table.
    let budget_bytes = ((memory_limit_mb * 1024 * 1024) / 3) as usize;

    // Decide partition count based on a rough right-side size estimate.
    // We peek at up to 1024 rows to estimate avg row size, keeping them buffered.
    const SAMPLE_SIZE: usize = 128;
    let mut sample: Vec<Row> = Vec::with_capacity(SAMPLE_SIZE);
    while sample.len() < SAMPLE_SIZE {
        match right_src.next_row()? {
            None => break,
            Some(row) => {
                if right_preds.iter().all(|p| compiled_predicate_matches(p, &row).unwrap_or(false)) {
                    sample.push(row);
                }
            }
        }
    }

    // Estimate total right size using sample avg * right row count heuristic.
    // Since we don't know total right count, use a conservative estimate.
    let sample_bytes: usize = sample.iter().map(|r| {
        r.values.iter().map(|v| match v {
            Data::String(s) => s.len() + 2,
            _ => 8,
        }).sum::<usize>() + 24
    }).sum();
    let avg_row_bytes = if sample.is_empty() { 64 } else { sample_bytes / sample.len() };
    let num_partitions = if sample.len() < SAMPLE_SIZE {
        let est_total = sample.len() * avg_row_bytes;
        if est_total <= budget_bytes {
            1usize
        } else {
            ((est_total / budget_bytes) + 1).max(2).min(256)
        }
    } else {
        // Stream limit hit: dataset could be massive.
        // Pre-allocate many partitions conservatively.
        256usize
    };

    let block_size = disk_client.borrow().block_size;

    // ------------------------------------------------------------------
    // Stream right source into scratch partitions (write buffer per part)
    // ------------------------------------------------------------------
    const WRITE_BUF: usize = SCRATCH_WRITE_BUF; // rows per partition before flushing to scratch
    let mut right_bufs: Vec<Vec<Row>> = (0..num_partitions).map(|_| Vec::new()).collect();
    let mut right_run_parts: Vec<Vec<ScratchRunMeta>> = (0..num_partitions).map(|_| Vec::new()).collect();

    // Helper: flush one partition buffer to scratch
    let flush_partition = |p: usize, bufs: &mut Vec<Vec<Row>>, run_parts: &mut Vec<Vec<ScratchRunMeta>>,
                           disk_client: &SharedDiskClient, scratch: &mut ScratchAllocator,
                           schema: &[DataType]| -> Result<()> {
        if bufs[p].is_empty() { return Ok(()); }
        let meta = write_run_to_scratch(&bufs[p], schema, disk_client.clone(), scratch)?;
        run_parts[p].push(meta);
        bufs[p].clear();
        Ok(())
    };

    // Process the already-sampled rows
    for row in sample {
        let p = partition_of(&row.values[right_key], num_partitions);
        right_bufs[p].push(row);
        if right_bufs[p].len() >= WRITE_BUF {
            flush_partition(p, &mut right_bufs, &mut right_run_parts, &disk_client, scratch, right_schema)?;
        }
    }

    // Process remaining rows from stream
    while let Some(row) = right_src.next_row()? {
        if !right_preds.iter().all(|p| compiled_predicate_matches(p, &row).unwrap_or(false)) {
            continue;
        }
        let p = partition_of(&row.values[right_key], num_partitions);
        right_bufs[p].push(row);
        if right_bufs[p].len() >= WRITE_BUF {
            flush_partition(p, &mut right_bufs, &mut right_run_parts, &disk_client, scratch, right_schema)?;
        }
    }
    // Flush remaining buffers
    for p in 0..num_partitions {
        flush_partition(p, &mut right_bufs, &mut right_run_parts, &disk_client, scratch, right_schema)?;
    }
    drop(right_bufs);

    // ------------------------------------------------------------------
    // Partition left side the same way
    // ------------------------------------------------------------------
    let mut left_bufs: Vec<Vec<Row>> = (0..num_partitions).map(|_| Vec::new()).collect();
    let mut left_run_parts: Vec<Vec<ScratchRunMeta>> = (0..num_partitions).map(|_| Vec::new()).collect();

    for row in left {
        let p = partition_of(&row.values[left_key], num_partitions);
        left_bufs[p].push(row.clone());
        if left_bufs[p].len() >= WRITE_BUF {
            if let Ok(meta) = write_run_to_scratch(&left_bufs[p], left_schema, disk_client.clone(), scratch) {
                left_run_parts[p].push(meta);
            }
            left_bufs[p].clear();
        }
    }
    for p in 0..num_partitions {
        if !left_bufs[p].is_empty() {
            if let Ok(meta) = write_run_to_scratch(&left_bufs[p], left_schema, disk_client.clone(), scratch) {
                left_run_parts[p].push(meta);
            }
            left_bufs[p].clear();
        }
    }
    drop(left_bufs);

    // ------------------------------------------------------------------
    // Join each partition pair in-memory
    // ------------------------------------------------------------------
    let mut output: Vec<Row> = Vec::new();

    for p in 0..num_partitions {
        // Load right partition (build side)
        let mut right_part: Vec<Row> = Vec::new();
        for run in &right_run_parts[p] {
            right_part.extend(load_scratch_run(run, right_schema, disk_client.clone(), block_size)?);
        }
        if right_part.is_empty() { continue; }

        // Load left partition (probe side)
        let mut left_part: Vec<Row> = Vec::new();
        for run in &left_run_parts[p] {
            left_part.extend(load_scratch_run(run, left_schema, disk_client.clone(), block_size)?);
        }
        if left_part.is_empty() { continue; }

        let joined = simple_hash_join(&left_part, left_key, &right_part, right_key);
        output.extend(joined);
    }

    Ok(output)
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
                    filter, cross, ctx, disk_client, scratch, memory_limit_mb, None,
                );
            }
            let child = execute_query_op(&filter.underlying, ctx, disk_client, scratch, memory_limit_mb)?;
            Ok(Box::new(FilterSource::new(filter, child)))
        }
        QueryOp::Project(project) => {
            if let QueryOp::Filter(filter) = project.underlying.as_ref() {
                if let QueryOp::Cross(cross) = filter.underlying.as_ref() {
                    let mut required_columns = HashSet::new();
                    collect_required_columns_for_project(project, &mut required_columns);
                    let child = execute_filter_cross(
                        filter,
                        cross,
                        ctx,
                        disk_client,
                        scratch,
                        memory_limit_mb,
                        Some(&required_columns),
                    )?;
                    return Ok(Box::new(ProjectSource::new(project, child)?));
                }
            }

            if let QueryOp::Sort(sort) = project.underlying.as_ref() {
                if let QueryOp::Filter(filter) = sort.underlying.as_ref() {
                    if let QueryOp::Cross(cross) = filter.underlying.as_ref() {
                        let mut required_columns = HashSet::new();
                        collect_required_columns_for_project(project, &mut required_columns);
                        collect_required_columns_for_sort(sort, &mut required_columns);
                        let child = execute_filter_cross(
                            filter,
                            cross,
                            ctx,
                            disk_client.clone(),
                            scratch,
                            memory_limit_mb,
                            Some(&required_columns),
                        )?;
                        let sorted =
                            external_sort(sort, child, disk_client, scratch, memory_limit_mb)?;
                        return Ok(Box::new(ProjectSource::new(project, sorted)?));
                    }
                }
            }

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
            let left = execute_query_op(&cross.left, ctx, disk_client.clone(), scratch, memory_limit_mb)?;
            let right = execute_query_op(&cross.right, ctx, disk_client.clone(), scratch, memory_limit_mb)?;
            out_of_core_cross(left, right, disk_client, scratch, memory_limit_mb)
        }
    }
}

fn write_result_to_monitor(
    mut result_source: Box<dyn RowSource>,
    monitor_out: &mut dyn Write,
) -> Result<()> {
    monitor_out.write_all(b"validate\n")?;

    let mut line = String::with_capacity(512);
    while let Some(row) = result_source.next_row()? {
        line.clear();
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
