import cv2
import os
import numpy as np
import csv
import multiprocessing
import time
import matplotlib.pyplot as plt

def scale_image(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)


def save_to_csv(file_path, data, caramelle_da_file, total_found, total_expected, photo_id):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['ID', 'Label', 'Numero Caramelle Trovate', 'Falsi Positivi', 'Falsi Negativi', 'Totale', 'Totale Aspettato', 'Differenza', 'Differenza totale','Caramelle Aspettate_per_Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Se il file è vuoto, scriv0 l'header
        if csvfile.tell() == 0:
            writer.writeheader()

        for label, count in data.items():
            confronto = "Corretto" if label in caramelle_da_file and count == caramelle_da_file[label] else "Errato"
            difference = abs(count - caramelle_da_file.get(label, 0)) if confronto == "Errato" else None
            differenza_totale = total_expected - total_found

            falsi_positivi = max(0, count - caramelle_da_file.get(label, 0))
            falsi_negativi = max(0, caramelle_da_file.get(label, 0) - count)

            writer.writerow({
                'ID': photo_id,
                'Label': label,
                'Numero Caramelle Trovate': count,
                'Falsi Positivi': falsi_positivi,
                'Falsi Negativi': falsi_negativi,
                'Totale': total_found,
                'Totale Aspettato': total_expected,
                'Differenza': difference,
                'Differenza totale': differenza_totale,
                'Caramelle Aspettate_per_Label': caramelle_da_file.get(label, 0)
            })

def perform_template_matching(image, template, label, threshold_small, threshold_large):
    found_matches = []
    height, width = image.shape[:2]
    scale_info = []

    if height < 500 or width < 500:
        print("Entro in piccolo")
        min_scale = 1.0
        max_scale = 2.0
        scale_step = 0.1
        threshold = threshold_small
    else:
        min_scale = 1.0
        max_scale = 0.2
        scale_step = -0.03
        threshold = threshold_large

    for scale in np.arange(min_scale, max_scale, scale_step):
        scaled_image = scale_image(image, scale)

        height, width = scaled_image.shape[:2]
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
                    found_matches.append((scaled_coords, label, scale, threshold))
                    scale_info.append(scale)
            except cv2.error as e:
                print(f"Error: {e}")

    return found_matches

def worker_process(template_info_tuple, image):
    filename, template_info = template_info_tuple
    label = template_info["label"]
    threshold_small = template_info["threshold_small"]
    threshold_large = template_info["threshold_large"]

    template_path = os.path.join("templates", filename)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if template is None:
        print(f"Error loading template: {template_path}")
        return []

    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        template = cv2.resize(template, (image.shape[1], image.shape[0]))

    matches = perform_template_matching(image, template, label, threshold_small, threshold_large)

    return matches

def main():

    image_path = "images/Validation/IMG_7695.PNG"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    photo_id = os.path.basename(image_path).split('.')[0]


    if image is None:
        print("Error loading the image.")
        return

    caramelle_utente = int(input("Inserisci il numero di caramelle: "))
    start_time = time.time()

    template_folder = "templates"
    template_info = {
        "blue_candy copia.png": {"label": "B",'threshold_small':0.8,"threshold_large":0.773},
        "bluee_trasp.png": {"label": "Bt", 'threshold_small': 0.8, "threshold_large": 0.8440000000000001},
        "blu_gela.png": {"label": "Bg", 'threshold_small': 0.7, "threshold_large":0.8670000000000001 },#0.8340000000000001
        "green_candy.png": {"label": "G",'threshold_small':0.75,"threshold_large":0.8280000000000001},
        "G_trasssoo.png": {"label": "Gt",'threshold_small':0.8,"threshold_large":0.9310000000000002},
        "Blue_strip_V.png":{"label":"bSV",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_l_G.png": {"label": "BLG",'threshold_small':0.7859999999999999,"threshold_large":0.9350000000000002},
        "blue_liqui_trans.png": {"label": "Blt",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_wrap.png": {"label": "Bw",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_wrap_crystal.png": {"label": "BwC",'threshold_small':0.7859999999999999,"threshold_large":0.8670000000000001},
        "blue_crystal.png": {"label": "BC",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_strip_liqui_transp.png": {"label": "BSLt",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blu_liqui_wrap_trasp.png": {"label": "bLWT",'threshold_small':0.7859999999999999,"threshold_large":0.8440000000000001},
        "blu_wrap_transp_liqui.png":{"label":"BwtL",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "blue_stii.png": {"label": "BStrip",'threshold_small':0.7859999999999999,"threshold_large":0.83},
        "blue_orizzontal_strip.png": {"label": "BOS",'threshold_small':0.7859999999999999,"threshold_large":0.8},
        "Glt.png": {"label": "GlT",'threshold_small':0.6519999999999999,"threshold_large":0.9490000000000002},
        "Gln.png": {"label": "GlN",'threshold_small':0.6519999999999999,"threshold_large":0.9240000000000002},
        "Glg.png": {"label": "GlG",'threshold_small':0.6519999999999999,"threshold_large":0.9130000000000001},

        "green_gela.png": {"label": "Gg",'threshold_small':0.85,"threshold_large":0.9000000000000001},
        "green_stri.png": {"label": "GS",'threshold_small':0.85,"threshold_large":0.7769999999999999},
        "gren_stri_gela.png": {"label": "GSG",'threshold_small':0.85,"threshold_large":0.83},
        "green_crystal.png": {"label": "GC",'threshold_small':0.85,"threshold_large":0.8180000000000001},
        "Green_wrap_cristal.png": {"label": "GWC",'threshold_small':0.85,"threshold_large":0.8},
        "green_wrap.png": {"label": "GW",'threshold_small':0.85,"threshold_large":0.781},
        "green_wrap_transp.png": {"label": "GWT",'threshold_small':0.85,"threshold_large":0.8},
        "green_oriz_strip.png":{"label": "GOS",'threshold_small':0.85,"threshold_large":0.8},
        "orange_candy.png": {"label": "O",'threshold_small':0.8,"threshold_large":0.8},
        "orange_trasp2.png": {"label": "Ot",'threshold_small':0.9360000000000002,"threshold_large":0.8740000000000001},
        "orange_gela.png": {"label": "Og",'threshold_small':0.8330000000000001,"threshold_large":0.8400000000000001},
        "orange_liqui_gray.png": {"label": "Olg",'threshold_small':0.796,"threshold_large":0.9380000000000002},
        "orange_liqui_trasp.png": {"label": "Olt",'threshold_small':0.8600000000000001,"threshold_large":0.8720000000000001},
        "Orange_wrap.png": {"label": "Ow",'threshold_small':0.8600000000000001,"threshold_large":0.8},
        "orange_wrap_transp.png": {"label": "OwT",'threshold_small':0.8600000000000001,"threshold_large":0.8390000000000001},
        "orange_crystal.png": {"label": "OC",'threshold_small':0.8600000000000001,"threshold_large":0.83},
        #"orange_strip.png": {"label": "OS",'threshold_small':0.8600000000000001,"threshold_large":0.83},
        "Orange_strip_O.png": {"label": "OSO",'threshold_small':0.8600000000000001,"threshold_large":0.83},
        "Orange_strip_vertical.png": {"label": "OSV",'threshold_small':0.8600000000000001,"threshold_large":0.8},
        "orange_liqui_stri.png": {"label": "OLS",'threshold_small':0.8600000000000001,"threshold_large":0.8},
        "Orange_wrap_liqui_n.png": {"label": "OwLN",'threshold_small':0.8600000000000001,"threshold_large":0.8},
        "purp_candy.png": {"label": "P",'threshold_small':0.762,"threshold_large":0.8},
        "P_gelatina.png": {"label": "Pg",'threshold_small':0.6309999999999999,"threshold_large":0.8210000000000001},
        "pur_tas.png":{"label": "Pt",'threshold_small': 0.76,"threshold_large": 0.8380000000000001},
        "purple_liq_gela.png":{"label": "PlG",'threshold_small': 0.76,"threshold_large":0.9020000000000001},
        "purple_liqui_trasp.png":{"label": "PlT",'threshold_small': 0.76,"threshold_large":0.9380000000000002},
        "purple_liqui_normal.png":{"label": "PlN",'threshold_small': 0.76,"threshold_large":0.9400000000000002},
        "purple_wrap.png":{"label": "Pw",'threshold_small': 0.76,"threshold_large":0.785},
        "purple_wrap_transp.png":{"label": "PwT",'threshold_small': 0.76,"threshold_large":0.83},
        "purp_strip_transp.png":{"label": "PST",'threshold_small': 0.76,"threshold_large":0.8570000000000001},
        "purp_crystal.png":{"label": "PC",'threshold_small': 0.76,"threshold_large":0.8},
        "Purple_strip_vertical.png":{"label":"PSV",'threshold_small': 0.76,"threshold_large":0.83},
        "purp_stri.png":{"label": "PS",'threshold_small': 0.76,"threshold_large":0.83},
        "purpl_strip_gela.png":{"label": "PSG",'threshold_small': 0.76,"threshold_large":0.83},
        "purple_liqui_wrap.png":{"label": "PLW",'threshold_small': 0.76,"threshold_large":0.759},
        "liq.png":{"label": "L",'threshold_small': 0.76,"threshold_large":0.8520000000000001},
        "liqui_transp.png":{"label": "Lt",'threshold_small': 0.76,"threshold_large":0.807},#0.8690000000000001
        "liqui_crystal.png":{"label": "LK",'threshold_small': 0.76,"threshold_large":0.756},
        "liqui_gela.png":{"label": "LG",'threshold_small': 0.76,"threshold_large":0.9060000000000001},
        "meringa_c.png":{"label": "MC",'threshold_small': 0.76,"threshold_large": 0.9220000000000002},
        "meringa_crystal.png":{"label": "MK",'threshold_small': 0.76,"threshold_large":0.8},
        "meringa_transp.png":{"label": "MT",'threshold_small': 0.76,"threshold_large":0.9490000000000002},
        "meringa_liqui.png":{"label": "MLi",'threshold_small': 0.76,"threshold_large":0.785},
        #"yellow_candy.png": {"label": "Y", "threshold_small": 0.9, "threshold_large": 0.95},
        "red_copiiaaa.png": {"label": "R",'threshold_small':0.8690000000000001,"threshold_large":0.8820000000000001},
        #"green_liqui_gray.png": {"label": "gcl", "threshold_small": 0.4, "threshold_large": 0.5359999999999998},
        #"purp_liqui_gela copia.png": {"label": "pcl", "threshold_small": 0.5, "threshold_large": 0.7},
        "red_tras.png": {"label": "Rt",'threshold_small':0.8340000000000001,"threshold_large":0.8490000000000001},
        "red_gela.png": {"label": "Rg",'threshold_small':0.8140000000000001,"threshold_large": 0.8640000000000001},
        "Red_strip.png": {"label": "RS",'threshold_small':0.8140000000000001,"threshold_large": 0.8640000000000001},
        "red_wrap.png": {"label": "Rw",'threshold_small':0.8,"threshold_large": 0.8180000000000001},
        "magic.png": {"label": "MA",'threshold_small':0.8,"threshold_large": 0.8180000000000001},
    }

    template_info_list = [(filename, template_info) for filename, template_info in template_info.items()]
    num_processes = min(multiprocessing.cpu_count(), len(template_info_list))
    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.starmap(worker_process, [(info, image) for info in template_info_list])

    pool.close()
    pool.join()


    all_points = [point for result in results for point in result]
    all_points.sort(key=lambda x: x[3], reverse=True)

    caramelle_per_label = {}
    matched_points = set()
    caramelle_label_position = {}

    for item in all_points:
        (x1, y1, x2, y2), label, scale, threshold = item

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        overlapping = False
        for point, existing_label, existing_scale, existing_threshold in caramelle_label_position.values():
            px1, py1, px2, py2 = point

            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                overlapping = True

                if threshold > existing_threshold:
                    caramelle_label_position.pop(point)
                    matched_points.remove(point)
                    caramelle_label_position[(x1, y1, x2, y2)] = ((x1, y1, x2, y2), label, scale, threshold)
                    matched_points.add((x1, y1, x2, y2))

                break

        if not overlapping:
            matched_points.add((x1, y1, x2, y2))
            caramelle_label_position[(x1, y1, x2, y2)] = ((x1, y1, x2, y2), label, scale, threshold)
            caramelle_per_label[label] = caramelle_per_label.get(label, 0) + 1
    scale_counts = {}
    for (x1, y1, x2, y2), label, scale, threshold in caramelle_label_position.values():
        # Draw rectangles on the image
        #print(scale)
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    caramelle_da_file = {}
    for line in open('images/Validation/IMG_7695.txt'):
        line = line.strip()
        if ':' in line:
            label, num_caramelle = line.split(':')
            label = label.strip()
            num_caramelle = int(num_caramelle.strip())
            caramelle_da_file[label] = num_caramelle

    # Confronto e stampa delle informazioni sulle label
    for label, cont in caramelle_per_label.items():
        print(f"Label: {label}, Caramelle trovate: {cont}")
        if label in caramelle_da_file:
            if cont == caramelle_da_file[label]:
                print(f"Il numero di caramelle per la label '{label}' coincide con il conteggio.")
            else:
                print(f"Attenzione: Il numero di caramelle per la label '{label}' non coincide. Hai trovato {cont}, ma il file dice {caramelle_da_file[label]}.")
        else:
            print(f"Attenzione: La label '{label}' trovata nel file non corrisponde a nessuna delle label rilevate nell'immagine.")

    total_found = len(matched_points)
    print(total_found)
    total_expected = sum(caramelle_da_file.values())

    csv_file_path = 'risultati.csv'
    header = ['ID', 'Label', 'Numero Caramelle Trovate', 'Confronto', 'Totale', 'Totale Aspettato']
    header_written = os.path.exists(csv_file_path)

    # Scrittura dell'header solo se il file non esiste
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not header_written:
            writer.writerow(header)

        # Aggiorna il CSV per i confronti corretti
    save_to_csv(csv_file_path, caramelle_per_label, caramelle_da_file, total_found, total_expected, photo_id)

    scale_values = list(scale_counts.keys())
    counts = list(scale_counts.values())
    print(scale_values)
    print(counts)
    
    plt.scatter(scale_values, counts, color='skyblue')
    plt.xscale('linear')
    plt.xlabel('Scale')
    plt.ylabel('Counts')
    plt.title('Distribuzione per Scala')
    plt.grid(True)
    plt.show()

    cv2.putText(image, str(len(matched_points)), (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0), 2)
    image_large = cv2.resize(image, (1000, 1200))
    cv2.imshow("Image with Matches", image_large)
    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
