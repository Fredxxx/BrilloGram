import glob
import json
import os
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import tifffile 

pathMain = r'C:\Fred\temp\90deg_08NA_gauss_32_00deg_3'
#pathMain = r'\\prevedel.embl.de\prevedel\members\Goerlitz\BrillouinMicroscopy\20251215_90degVS180deg\90deg'


Lfolder = ['sys', 'det', 'exc']
Lname = ['System', 'Detection', 'Excitation']
deg = '90deg'
dn = '0.02dn'  # Falls "dn" im Titel enthalten sein soll


numFolders = len(Lfolder)
for i, (nFold, nTitle) in enumerate(zip(Lfolder, Lname)):
    nameFold = nFold
    titleMain = f"{nTitle} - {deg} - {dn}"
    
    # Pfad erstellen und JSON-Dateien suchen
    search_path = os.path.join(pathMain, nameFold, '*.json')
    json_files = glob.glob(search_path)
    

#nameFold = 'sys'
#titleMain = 'System - 180deg - 0.1dn'
#json_files = glob.glob(os.path.join(pathMain, nameFold, '*.json'))

    data = []
    with open(json_files[-1], 'r') as f:
        try:
            # Parse the JSON file into a Python dictionary
            res = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
            s = [res.x + 1, res.y + 1, res.z + 1]
        except json.JSONDecodeError:
            print(f"Failed to parse {json_files[-1]}")
    
    s = [32, 32, 1]#[64, 64, 1]
    thCOMdeg = np.zeros((s))
    thCOMbs = np.zeros((s))
    thSTDdeg = np.zeros((s))
    thSTDbs = np.zeros((s))
    
    numFiles = len(json_files)
    
    #for p in json_files:
    for j, p in enumerate(json_files, 1):
        with open(p, 'r') as f:
            try:
                res = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
                perc = ((i + (j / numFiles)) / numFolders) * 100
                print(f"progress...   {perc:6.2f}%")
                thCOMdeg[res.x, res.y, res.z] = res.thetaCOM
                thCOMbs[res.x, res.y, res.z] = res.thetaComBS
                thSTDdeg[res.x, res.y, res.z] = res.thetaSTD
                thSTDbs[res.x, res.y, res.z] = res.thetaStdBS
            except json.JSONDecodeError:
                print(f"Failed to parse {f}")
    
    
    #%%
    # Pfade definieren
    plot_path = os.path.join(pathMain, "results")
    os.makedirs(plot_path, exist_ok=True)
    
    # Daten-Struktur (Titel, Array-Variable, Basis-Name, Einheit)
    plot_data = [
        ("Brillouin Shift (COM)", np.squeeze(thCOMdeg), "thCOMdeg", "deg"),
        ("Brillouin Shift (COM)", np.squeeze(thCOMbs), "thCOMbs", "GHz"),
        ("Brillouin Width (STD)", np.squeeze(thSTDdeg), "thSTDdeg", "deg"),
        ("Brillouin Width (STD)", np.squeeze(thSTDbs), "thSTDbs", "GHz")
    ]
    
    # save as tiff and txt
    for title, data, name, unit in plot_data:
        txt_name = os.path.join(plot_path, f"{nameFold}_{name}.txt")
        np.savetxt(txt_name, data, delimiter='\t', header=f"{title} [{unit}]")
        tif_name = os.path.join(plot_path, f"{nameFold}_{name}.tif")
        tifffile.imwrite(tif_name, data.astype(np.float32))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle(titleMain, fontsize=20, fontweight='bold', y=0.98)
    
    for i, (title, data, name, unit) in enumerate(plot_data):
        ax = axes[i]
        im = ax.imshow(data, cmap='viridis', interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(unit)
        ax.set_title(f"{title} [{unit}]")
        ax.set_xlabel("x-index")
        ax.set_ylabel("y-index")
    
    plt.tight_layout()
    combined_png = os.path.join(plot_path, f"{nameFold}_fig.png")
    plt.savefig(combined_png, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    


