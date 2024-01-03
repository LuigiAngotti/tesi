import cv2
import os
import numpy as np
from imutils.object_detection import non_max_suppression

# Funzione per ridimensionare un'immagine
def scale_image(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)

# Funzione per eseguire il riconoscimento dei template
def perform_template_matching(image, template, label, threshold):
    found_matches = []

    height, width = image.shape[:2]
    #print(height)
    #print(width)
    if height < 1000 or width < 1000:
        min_scale = 1.0
        max_scale = 2.0
        scale_step = 0.1
    else:
        min_scale = 1.0
        max_scale = 0.2
        scale_step = -0.03
        #print("grande")

    for scale in np.arange(min_scale, max_scale, scale_step):
        scaled_image = scale_image(image, scale)
        if scaled_image is not None and template is not None:
            try:
                res = cv2.matchTemplate(scaled_image, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)

                for pt in zip(*loc[::-1]):
                    x1, y1 = pt
                    x2, y2 = x1 + template.shape[1], y1 + template.shape[0]
                    scaled_coords = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
                    found_matches.append((scaled_coords, label, scale))
                    #cv2.rectangle(scaled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #cv2.putText(scaled_image, f"{label} (Scale {scale:.1f})", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    #cv2.imshow("Resized Image with Matching", scaled_image)
                    #cv2.waitKey(100)

            except cv2.error as e:
                print(f"Error: {e}")

    return found_matches

# Leggi il file delle annotazioni
annotations = {}
with open("annotazioni.txt", "r") as file:
    for line in file:
        parts = line.strip().split(":")
        #print(parts)
        if len(parts) == 2:
            label = parts[0].strip()
            count = int(parts[1].strip())
            annotations[label] = count


def main():
    image = cv2.imread("images/png/Screenshot_20231107_204016_Candy Crush Saga.png", cv2.IMREAD_COLOR)

    max_image_width = 1000
    if image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    all_points = []
    caramelle_dict = {label: [] for label in annotations}

    template_folder = "templates"
    template_info = {
        "blue_candy copia.png": {"label": "B"},
        "bl_trasp.png": {"label": "Bt"},
        "blu_gela.png": {"label": "Bg"},
        "blue_crystal.png": {"label": "BC"},
        "blue_liqui_trans.png": {"label": "Blt"},
        "blue_l_G.png":{"label":"blg"},
        "blue_wrap.png":{"label":"bw"},
        "blue_wrap_crystal.png":{"label":"bwC"},
        "blu_liqui_wrap_trasp.png":{"label":"blWT"},
        "blu_wrap_transp_liqui.png":{"label":"BwtL"},
        "blue_stii.png":{"label":"bS"},
        "blue_strip_liqui_transp.png":{"label":"bSlT"},
        "green_candy.png": {"label": "G"},
        "G_trasssoo.png": {"label": "Gt"},
        "green_gela.png": {"label": "Gg"},
        "gren_candy_liqui_grey copia.png": {"label": "Glg"},
        "green_liqui_trasp.png": {"label": "Glt"},
        "green_liqui_normal.png": {"label": "Gln"},
        "green_crystal.png": {"label": "GC"},
        "green_stri.png": {"label": "GS"},
        "gren_stri_gela.png": {"label": "GSG"},
        "green_wrap_transp.png": {"label": "GWT"},
        "green_wrap.png": {"label": "GW"},
        "orange_candy.png": {"label": "O"},
        "orange_trasp2.png": {"label": "Ot"},
        "orange_gela.png": {"label": "Og"},
        "orange_liqui_gray.png": {"label": "Olg"},
        "orange_liqui_trasp.png": {"label": "Olt"},
        "orange_wrap.png": {"label": "Ow"},
        "orange_wrap_transp.png": {"label": "OwT"},
        "orange_crystal.png": {"label": "OC"},
        "orange_strip.png": {"label": "OS"},
        "orange_liqui_stri.png": {"label": "OLS"},
        "purp_candy.png": {"label": "P"},
        "P_gelatina.png": {"label": "Pg"},
        "pur_tas.png":{"label": "Pt"},
        "purple_liq_gela.png":{"label":"PlG"},
        "purple_liqui_trasp.png":{"label":"PlT"},
        "purple_liqui_normal.png":{"label":"PlN"},
        "purple_wrap.png":{"label":"Pw"},
        "purple_wrap_transp.png":{"label":"PwT"},
        "purp_crystal.png":{"label":"PC"},
        "purp_stri.png":{"label":"PS"},
        "purpl_strip_gela.png":{"label":"PSG"},
        "purple_liqui_wrap.png":{"label":"PLW"},


        #"yellow_candy.png": {"label": "Y", "threshold_small": 0.9, "threshold_large": 0.95},
        "red_copiiaaa.png": {"label": "R"},
        "green_liqui_gray.png": {"label": "gcl"},

        "red_tras.png": {"label": "Rt"},
        "red_gela.png": {"label": "Rg"},
        "red_liqui_gray.png": {"label": "rlg"},
        "liq.png": {"label": "L"},
        "liqui_transp.png": {"label": "Lt"},
        "liqui_gela.png": {"label": "LG"},
        "liqui_crystal.png": {"label": "LK"},
        "meringa_c.png": {"label": "MC"},
        "meringa_crystal.png": {"label": "MK"},
        "meringa_transp.png": {"label": "MT"},
        "meringa_liqui.png": {"label": "ML"},


    }

    all_points = []
    caramelle_dict = {label: [] for label in annotations}  # Dizionario per memorizzare i punti per etichetta

    for filename in os.listdir(template_folder):
        if filename.endswith(".png") and filename in template_info and template_info[filename]["label"] in annotations:
            template_path = os.path.join(template_folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                print(f"Errore nel caricamento del template: {template_path}")
                continue
            label = template_info[filename]["label"]
            target_count = annotations.get(label, 0)
            current_threshold = 0.8
            temp_label = ""

            while True:
                found_matches = perform_template_matching(image, template, label, current_threshold)
                caramelle_trovate = 0

                for item in found_matches:
                    (x1, y1, x2, y2), label, scale = item
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if temp_label != label:
                        print("ciao")
                        caramelle_trovate = 0
                        temp_label = label

                    is_overlapping = False
                    for existing_item in caramelle_dict[label]:
                        ex_x1, ex_y1, ex_x2, ex_y2, _, _ = existing_item
                        if (x1 < ex_x2 and x2 > ex_x1 and y1 < ex_y2 and y2 > ex_y1):
                            is_overlapping = True
                            break

                    if not is_overlapping:
                        caramelle_dict[label].append((x1, y1, x2, y2, label, scale))
                        caramelle_trovate += 1

                #print(label)
                print(target_count)
                print(caramelle_trovate)
                print(current_threshold)

                if caramelle_trovate == target_count:
                    print("esco")
                    print(label)
                    print(current_threshold)
                    with open("thresholds.txt", "a") as file:
                        file.write(f"{label}: {current_threshold}\n")

                    break
                elif caramelle_trovate < target_count:
                    #print("minore")
                    caramelle_dict[label].clear()
                    current_threshold -= 0.001
                elif caramelle_trovate > target_count:
                    #print("maggiore")
                    caramelle_dict[label].clear()
                    current_threshold += 0.001


                if current_threshold < 0.5 or current_threshold > 0.95:
                    break

    # Creiamo una copia dell'immagine originale per disegnare i rettangoli
    image_with_rectangles = image.copy()

    for label, points in caramelle_dict.items():
        for item in points:
            x1, y1, x2, y2, _, _ = item
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image_with_rectangles, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_rectangles, f"{label} ", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    caramelle_trovate = sum(len(points) for points in caramelle_dict.values())

    cv2.putText(image_with_rectangles, str(caramelle_trovate), (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 4)
    image_large = cv2.resize(image_with_rectangles, (1000, 2024))
    cv2.imshow("Image with Matches", image_large)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
