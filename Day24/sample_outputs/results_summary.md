# OCR results on all sample images

Numbers are EasyOCR's average confidence (1.00 = completely sure).

| Image | none | light | contrast | scanned | Best |
|---|---|---|---|---|---|
| `book_page.png` | 0.79 | 0.81 | **0.81** | 0.75 | contrast |
| `document_clean.png` | 0.77 | 0.80 | **0.87** | 0.76 | contrast |
| `document_dim_lighting.png` | 0.84 | 0.84 | **0.86** | 0.77 | contrast |
| `document_low_resolution.png` | **0.77** | 0.75 | 0.72 | 0.23 | none |
| `document_rotated.png` | 0.94 | **0.94** | 0.94 | 0.92 | light |
| `document_small_text.png` | 0.87 | 0.84 | **0.88** | 0.09 | contrast |
| `form_enrolment.png` | 0.87 | **0.91** | 0.85 | 0.85 | light |
| `form_low_contrast.png` | 0.84 | 0.84 | 0.88 | **0.89** | scanned |
| `form_with_checkboxes.png` | 0.86 | **0.92** | 0.90 | 0.84 | light |
| `handwritten_note.png` | 0.84 | 0.78 | **0.85** | 0.67 | contrast |
| `id_card.png` | **0.92** | 0.89 | 0.91 | 0.76 | none |
| `invoice_clean.png` | 0.93 | 0.93 | **0.94** | 0.83 | contrast |
| `invoice_scanned_tint.png` | 0.94 | 0.94 | **0.94** | 0.85 | contrast |
| `invoice_skewed.png` | **0.92** | 0.90 | 0.90 | 0.73 | none |
| `receipt_clean.png` | **0.88** | 0.87 | 0.86 | 0.84 | none |
| `receipt_faded.png` | 0.75 | 0.73 | 0.78 | **0.83** | scanned |
| `receipt_noisy_lowlight.png` | **0.89** | 0.87 | 0.48 | 0.01 | none |
| `receipt_tiny_text.png` | 0.41 | 0.48 | **0.53** | 0.49 | contrast |

## Average confidence per preset (all images)

| Preset | Average confidence | Times it won |
|---|---|---|
| none | 0.835 | 5 |
| light | 0.836 | 3 |
| contrast | 0.828 | 8 |
| scanned | 0.672 | 2 |

## Deskew on vs off

Watch the line count, not the confidence - see README.

| Image | Deskew | Confidence | Lines | Chars | Corrected |
|---|---|---|---|---|---|
| `document_rotated.png` | off | 0.94 | 49 | 409 | 0.0° |
| `document_rotated.png` | on | 0.87 | 13 | 411 | -6.0° |
| `invoice_skewed.png` | off | 0.90 | 61 | 418 | 0.0° |
| `invoice_skewed.png` | on | 0.89 | 33 | 406 | 7.0° |
| `document_clean.png` | off | 0.87 | 13 | 408 | 0.0° |
| `document_clean.png` | on | 0.87 | 13 | 408 | 0.0° |
