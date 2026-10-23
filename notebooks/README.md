
# Notebook Resources

The original training notebook is `devanagari_cnn.ipynb`.

Keep the original notebook unchanged unless a deliberate training correction is required. The backend should load the exported model checkpoint rather than executing the entire notebook during application startup.

## Required model artifact

Filename: `devanagari_cnn_best.pth`

Expected location in the project:

`backend/models/devanagari_cnn_best.pth`

The checkpoint should contain:

- `model_state_dict`
- `num_classes`
- `class_names`
- `test_accuracy`
- `history`

The saved `class_names` order is critical because it defines the meaning of the model's output indices. The backend must use this saved mapping when interpreting predictions.

**Do not substitute a separately ordered character-label list for the training class mapping.** Doing so can cause valid predictions to be mapped to incorrect characters.

## Export procedure

1. Open `devanagari_cnn.ipynb` in the original notebook environment.
2. Run the notebook through the cell that saves `devanagari_cnn_best.pth`.
3. Download the generated checkpoint.
4. Place the file at `backend/models/devanagari_cnn_best.pth`.
5. Start the backend and verify that the checkpoint loads successfully.

Do not rerun training just to start the API when a valid checkpoint already exists.

## Backend integration

The FastAPI backend loads the checkpoint through its OCR service. The notebook is a development and training resource, not an application startup dependency.

Model weights should remain outside Git. The checkpoint can be stored locally at the required path while remaining excluded by the repository's `.gitignore`.

## OCR limitations

The current CNN is an isolated-character classifier with 46 output classes. Loading the checkpoint does not automatically provide complete page-level handwriting OCR.

Full-document recognition requires character segmentation and reconstruction or a separate document OCR implementation. Preserve this distinction when describing the application's current capabilities.
