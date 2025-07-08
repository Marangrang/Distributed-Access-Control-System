# Delta Sync Protocol

Each edge site runs a pull-only delta sync every 10 minutes to keep its local store up to date with the cloud. The sync process:

1. Reads a local checkpoint (last sync timestamp).
2. Requests only new or updated embeddings and thumbnails from the cloud since the last checkpoint.
3. Applies upserts locally: new or changed embeddings update the FAISS index, and new/updated thumbnails overwrite local copies.
4. Updates the checkpoint to the latest timestamp received from the cloud.

This approach minimizes bandwidth, ensures each site is always as current as possible, and tolerates intermittent connectivity. Full-resolution originals are never synced to the edge. Only the cloud can access and update the master dataset. 