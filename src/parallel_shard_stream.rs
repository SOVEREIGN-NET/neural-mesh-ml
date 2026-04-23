//! Multi-channel QUIC shard compression

use crate::Result;

/// Compressed shard for streaming.
#[derive(Debug, Clone)]
pub struct CompressedShard {
    pub shard_id: [u8; 32],
    pub data: Vec<u8>,
}

/// Parallel shard compression.
pub async fn parallel_shard_compress(shards: &[Vec<u8>]) -> Result<Vec<CompressedShard>> {
    // Stub: passthrough
    use blake3::hash;
    let compressed = shards
        .iter()
        .map(|data| CompressedShard {
            shard_id: hash(data).into(),
            data: data.clone(),
        })
        .collect();
    Ok(compressed)
}

/// Parallel shard decompression.
pub async fn parallel_shard_decompress(shards: &[CompressedShard]) -> Result<Vec<Vec<u8>>> {
    // Stub: extract data
    Ok(shards.iter().map(|s| s.data.clone()).collect())
}
