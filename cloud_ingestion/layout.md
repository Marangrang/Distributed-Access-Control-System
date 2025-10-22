# Object Storage Layout

## Originals
- Path: `originals/{driver_id}.jpg`
- Description: Full-resolution driver images, used for audit and retraining only. Not downloaded to edge sites.

## Thumbnails
- Path: `thumbnails/{driver_id}.jpg`
- Description: 112x112 pixel thumbnails for guard review at edge sites. Downloaded and synced to each site.

## Embeddings
- Stored in: Cloud vector store (e.g., PostgreSQL+pgvector)
- Key: `driver_id` (matches image and thumbnail naming)

This layout ensures efficient access and minimal bandwidth for edge devices, while the cloud retains all originals for compliance and retraining. 