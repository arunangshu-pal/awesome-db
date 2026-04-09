pub struct ScratchAllocator {
    next_block: u64,
    block_size: usize,
}

impl ScratchAllocator {
    pub fn new(anon_start: u64, block_size: usize) -> Self {
        Self { next_block: anon_start, block_size }
    }

    /// Allocates `num_blocks` contiguous blocks, returns starting block ID.
    pub fn alloc(&mut self, num_blocks: u64) -> u64 {
        let start = self.next_block;
        self.next_block += num_blocks;
        start
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
}