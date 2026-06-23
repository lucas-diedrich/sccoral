from scvi.dataloaders import DataSplitter


class ScCoralDataSplitter(DataSplitter):
    """`DataSplitter` that avoids singleton final training batches.

    ``BatchNorm`` raises a ``ValueError`` ("Expected more than 1 value per
    channel when training") when it receives a batch with a single sample. This
    happens whenever ``n_train_cells % batch_size == 1`` and the last minibatch
    of an epoch ends up with exactly one cell.

    To avoid this, the training dataloader drops the last batch *only* when it
    would be a singleton, so at most one cell is discarded and every other
    dataset shape behaves exactly as in the parent `DataSplitter`. Validation
    and test loaders are left untouched: they run in eval mode where ``BatchNorm``
    uses running statistics and accepts size-1 batches.
    """

    def train_dataloader(self):
        """Create train data loader, dropping a singleton final batch."""
        batch_size = self.data_loader_kwargs.get("batch_size", 128)
        # Only drop the last batch when it would contain a single cell.
        drop_last = len(self.train_idx) % batch_size == 1

        return self.data_loader_cls(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=drop_last,
            load_sparse_tensor=self.load_sparse_tensor,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )
