import os
import fitz  # PyMuPDF
from PIL import Image

# --- Config ---
pdf_folder = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/optimisation/test/toy_histograms'               # Folder with one-page PDF files
output_gif = 'toy_hists.gif'
duration_per_frame = 500

images = []

# --- Convert each PDF to image using PyMuPDF ---
for filename in sorted(os.listdir(pdf_folder)):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # load first page
        pix = page.get_pixmap(dpi=600)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

# --- Save as animated GIF ---
if images:
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration_per_frame,
        loop=0
    )
    print(f"GIF saved as {output_gif}")
else:
    print("No PDFs found.")