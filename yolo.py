import cv2
import os
import numpy as np
import time

# Funzione per ridimensionare un'immagine


def scale_image(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)

def scale_image2(image, max_width):
    print("ws")
    if image.shape[1] > max_width:
        scale_factor = max_width / image.shape[1]
        print(scale_factor)
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    else:
        return image

# Funzione per eseguire il riconoscimento dei template
def perform_template_matching(image, template, label, threshold_small, threshold_large):
    found_matches = []
    height, width = image.shape[:2]
    #print(height)
    #print(width)
    if height<500 or width<500  :
        print("piccola")
        min_scale = 1.0
        max_scale = 2.0
        scale_step = 0.1
        print("piccola")
        threshold = threshold_small
    else:
        min_scale = 1.0
        max_scale = 0.2
        scale_step = -0.03
        threshold = threshold_large



    for scale in np.arange(min_scale, max_scale, scale_step):
        print(scale)
        scaled_image = scale_image(image, scale)


        height, width = scaled_image.shape[:2]
        #print(height)
        #print(width)
        if scale >= 1.3:


            break
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

def main():
    image = cv2.imread("images/png/60BD3A25-CEC5-48EE-A465-E2F37A8AA137.png", cv2.IMREAD_COLOR)
    #image=cv2.resize(image,(1000,2000))




    # t=time.time()
    max_image_width = 1000
    if image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    #image = scale_image(image, max_image_width)
    print(image.shape[:2])
    caramelle_utente = int(input("Inserisci il numero di caramelle: "))
    #image = cv2.resize(image,(752,680))
    #print(image.shape[:2])


    template_folder = "templates"
    template_info = {
        "blue_candy copia.png": {"label": "B",'threshold_small':0.8,"threshold_large":0.773},
        "bluee_trasp.png": {"label": "Bt",'threshold_small':0.8,"threshold_large":0.8},
        "blu_gela.png": {"label": "Bg",'threshold_small':0.7,"threshold_large":0.805},
        "green_candy.png": {"label": "G",'threshold_small':0.75,"threshold_large":0.812},
        "G_trasssoo.png": {"label": "Gt",'threshold_small':0.8,"threshold_large":0.8790000000000001},
        "blue_l_G.png": {"label": "BLG",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_liqui_trans.png": {"label": "Blt",'threshold_small':0.7859999999999999,"threshold_large":0.6},
        "blue_wrap.png": {"label": "Bw",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_wrap_crystal.png": {"label": "BwC",'threshold_small':0.7859999999999999,"threshold_large":0.8610000000000001},
        "blue_crystal.png": {"label": "BC",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_strip_liqui_transp.png": {"label": "BSLt",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blu_liqui_wrap_trasp.png": {"label": "bLWT",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blu_wrap_transp_liqui.png":{"label":"BwtL",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_stii.png": {"label": "BStrip",'threshold_small':0.7859999999999999,"threshold_large":0.83},
        "green_liqui_trasp.png": {"label": "GlT",'threshold_small':0.6519999999999999,"threshold_large":0.7839999999999999},
        "green_liqui_normal.png": {"label": "GlN",'threshold_small':0.6519999999999999,"threshold_large":0.848},
        "gren_candy_liqui_grey copia.png": {"label": "GlG",'threshold_small':0.6519999999999999,"threshold_large":0.53},

        "green_gela.png": {"label": "Gg",'threshold_small':0.85,"threshold_large":0.8980000000000001},
        "green_stri.png": {"label": "GS",'threshold_small':0.85,"threshold_large":0.7769999999999999},
        "gren_stri_gela.png": {"label": "GSG",'threshold_small':0.85,"threshold_large":0.83},
        "green_crystal.png": {"label": "GC",'threshold_small':0.85,"threshold_large":0.8},
        "green_wrap.png": {"label": "GW",'threshold_small':0.85,"threshold_large":0.781},
        "green_wrap_transp.png": {"label": "GWT",'threshold_small':0.85,"threshold_large":0.8},
        "orange_candy.png": {"label": "O",'threshold_small':0.8,"threshold_large":0.8},
        "orange_trasp2.png": {"label": "Ot",'threshold_small':0.9360000000000002,"threshold_large":0.8670000000000001},
        "orange_gela.png": {"label": "Og",'threshold_small':0.8330000000000001,"threshold_large":0.8390000000000001},
        "orange_liqui_gray.png": {"label": "Olg",'threshold_small':0.796,"threshold_large":0.7830000000000003},
        "orange_liqui_trasp.png": {"label": "Olt",'threshold_small':0.8600000000000001,"threshold_large":0.724},
        "orange_wrap.png": {"label": "Ow",'threshold_small':0.8600000000000001,"threshold_large":0.693},
        "orange_wrap_transp.png": {"label": "OwT",'threshold_small':0.8600000000000001,"threshold_large":0.83},
        "orange_crystal.png": {"label": "OC",'threshold_small':0.8600000000000001,"threshold_large":0.83},
        "orange_strip.png": {"label": "OC",'threshold_small':0.8600000000000001,"threshold_large":0.83},
        "orange_liqui_stri.png": {"label": "OLS",'threshold_small':0.8600000000000001,"threshold_large":0.8},
        "purp_candy.png": {"label": "P",'threshold_small':0.762,"threshold_large":0.8},
        "P_gelatina.png": {"label": "Pg",'threshold_small':0.6309999999999999,"threshold_large":0.8210000000000001},
        "pur_tas.png":{"label": "Pt",'threshold_small': 0.76,"threshold_large": 0.8380000000000001},
        "purple_liq_gela.png":{"label": "PlG",'threshold_small': 0.76,"threshold_large":0.7689999999999999},
        "purple_liqui_trasp.png":{"label": "PlT",'threshold_small': 0.76,"threshold_large":0.815},
        "purple_liqui_normal.png":{"label": "PlN",'threshold_small': 0.76,"threshold_large":0.7939999999999999},
        "purple_wrap.png":{"label": "Pw",'threshold_small': 0.76,"threshold_large":0.785},
        "purple_wrap_transp.png":{"label": "PwT",'threshold_small': 0.76,"threshold_large":0.83},
        "purp_crystal.png":{"label": "PC",'threshold_small': 0.76,"threshold_large":0.8},
        "purp_stri.png":{"label": "PS",'threshold_small': 0.76,"threshold_large":0.83},
        "purpl_strip_gela.png":{"label": "PSG",'threshold_small': 0.76,"threshold_large":0.83},
        "purple_liqui_wrap.png":{"label": "PLW",'threshold_small': 0.76,"threshold_large":0.759},
        "liq.png":{"label": "L",'threshold_small': 0.76,"threshold_large":0.830000000000001},
        "liqui_transp.png":{"label": "Lt",'threshold_small': 0.76,"threshold_large":0.9030000000000001},
        "liqui_crystal.png":{"label": "LK",'threshold_small': 0.76,"threshold_large":0.756},
        "liqui_gela.png":{"label": "LG",'threshold_small': 0.76,"threshold_large":0.9440000000000002},
        "meringa_c.png":{"label": "MC",'threshold_small': 0.76,"threshold_large": 0.9140000000000001},
        "meringa_crystal.png":{"label": "MK",'threshold_small': 0.76,"threshold_large":0.8},
        "meringa_transp.png":{"label": "MT",'threshold_small': 0.76,"threshold_large":0.8},
        "meringa_liqui.png":{"label": "MT",'threshold_small': 0.76,"threshold_large":0.785},
        #0.795
        #0.9430000000000001
        #"yellow_candy.png": {"label": "Y", "threshold_small": 0.9, "threshold_large": 0.95},
        "red_copiiaaa.png": {"label": "R",'threshold_small':0.8690000000000001,"threshold_large":0.8690000000000001},
        #"green_liqui_gray.png": {"label": "gcl", "threshold_small": 0.4, "threshold_large": 0.5359999999999998},
        #"purp_liqui_gela copia.png": {"label": "pcl", "threshold_small": 0.5, "threshold_large": 0.7},
        "red_tras.png": {"label": "Rt",'threshold_small':0.8340000000000001,"threshold_large":0.9150000000000001},
        "red_gela.png": {"label": "Rg",'threshold_small':0.8140000000000001,"threshold_large": 0.8640000000000001},

    }

    all_points = []

    for filename in os.listdir(template_folder):
        if filename.endswith(".png") and filename in template_info:
            template_path = os.path.join(template_folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                print(f"Errore nel caricamento del template: {template_path}")
                continue
            label = template_info[filename]["label"]
            #threshold = template_info[filename]["threshold"]
            threshold_small = template_info[filename]["threshold_small"]
            threshold_large = template_info[filename]["threshold_large"]

            # Assicurati che le dimensioni del template siano minori o uguali a quelle dell'immagine principale
            if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                template = cv2.resize(template, (image.shape[1], image.shape[0]))

            found_matches = perform_template_matching(image, template, label, threshold_small, threshold_large)
            all_points.extend(found_matches)

    matched_points = set()
    #print("Dimensioni immagine:", image.shape)

    for item in all_points:
        (x1, y1, x2, y2), label, scale = item
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        overlapping = False
        for point in matched_points:
            px1, py1, px2, py2 = point
            # Verifica se c'è una sovrapposizione significativa
            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                overlapping = True
                break

        if not overlapping:
            matched_points.add((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    caramelle_trovate = len(matched_points)
    if caramelle_utente == caramelle_trovate:
        print("Hai trovato tutte le caramelle correttamente!")
    else:
        print(f"Hai trovato {caramelle_trovate} caramelle, ma avevi inserito {caramelle_utente}.")

    cv2.putText(image, str(len(matched_points)), (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0), 2)
    image_large = cv2.resize(image, (1000, 1200))
    cv2.imshow("Image with Matches", image_large)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
