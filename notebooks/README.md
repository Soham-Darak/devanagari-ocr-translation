
# Notebook Resources

The original training notebook is `devanagari_cnn.ipynb`.

Keep the original notebook unchanged unless a deliberate training correction
is required. The backend should load the exported checkpoint rather than
executing the entire notebook at application startup.

## Required model artifact

`devanagari_cnn_best.pth`

The checkpoint should contain:

- `model_state_dict`
- `num_classes`
- `class_names`
- `test_accuracy`
- `history`

The saved `class_names` order is critical. It must be used to interpret
the model's output indices.

Do not use a separately ordered character-label list as a substitute for
the training class mapping.

## Export procedure

Run the notebook through the cell that saves `devanagari_cnn_best.pth`.
Download that checkpoint from the notebook environment and place it in
`backend/models/`.

Do not rerun training merely to start the API if a valid checkpoint
already exists.
