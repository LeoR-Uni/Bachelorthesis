import os
import sys
maindir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(maindir_path)
import re
from edge_pydb import EdgeTable
import edge_pydb.util as edgeutil
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io import ascii
import csv
from itertools import combinations
from tqdm import tqdm
import pandas as pd
from edge_pydb.plotting import gridplot
from typing import Tuple
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
import plotly.graph_objects as go
import pandas as pd
import math as math
from astroHOG_master.astrohog import HOGcorr_frame
from astroHOG_master.astrohog2d import HOGcorr_imaLITE
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


marker = ['flux_Hbeta_sm', 'flux_[NII]6583_sm', 'flux_[SII]6731_sm', 'flux_Halpha_sm', 'flux_[OI]6300_sm', 'flux_[OIII]5007_sm', 'flux_AHa_corr_sm', 'flux_[OII]3727_sm', 'flux_[NII]6548_sm', 'flux_[SII]6717_sm', 'mom0_12', 'flux_[OIII]4959_sm']
flux_marker = ['flux_Hbeta_sm', 'flux_[NII]6583_sm', 'flux_[SII]6731_sm', 'flux_Halpha_sm', 'flux_[OI]6300_sm', 'flux_[OIII]5007_sm', 'flux_AHa_corr_sm', 'flux_[OII]3727_sm', 'flux_[NII]6548_sm', 'flux_[SII]6717_sm', 'flux_[OIII]4959_sm']
pos_marker = ["Name", "ix", "iy", "ra_abs", "dec_abs", "ra_off", "dec_off", "rad_arc", "azi_ang"]
def get_galaxyTable(galaxy):
    """
    Create a .dat file of the Galaxy-Data.

    Parameters:
    galaxy (str): The name of the galaxy you want to look up.
    use_data (str): Specify if you want to use all Pixels or reduced Data.

    Outputs:
    Saves a .dat file with relevant galaxy data.
    """
    


    Table_dir_Path = os.path.join(maindir_path,"Galaxy_Tables")

    if os.path.isdir(Table_dir_Path): 
        print("directory available.")
    else:
        print("TeSt")
        os.mkdir(Table_dir_Path)

    file_name = galaxy+"_table.dat"
    check_file = os.path.join(Table_dir_Path, file_name)

    # Check if the file already exists
    if os.path.isfile(check_file):
        print("The file for this galaxy already exists. The creation process will be skipped.")
        pass
        return

    # Convert galaxy name to UTF-8 format for use in Database
    galaxy_utf8 = galaxy.encode("utf-8")

    # Generate the Tables of the data you want to look at
    gal_tab = EdgeTable("edge_carma_allpix.2d_smo7.hdf5", path = "flux_elines_sm", cols=pos_marker+flux_marker)
    co_tab = EdgeTable("edge_carma_allpix.2d_smo7.hdf5", path = "comom_smo", cols=pos_marker+["mom0_12"])
    
    #specify the coordinate (signature) data like name, position etc.
    gal_tab.join(co_tab, keys=pos_marker)
    
    #isolate the galaxy you want to look at
    gal_tab = gal_tab.table[gal_tab["Name"]==galaxy_utf8]
   
    ascii.write(gal_tab, check_file, overwrite = True)





def ecHOG(marker1 = None, marker2 = None, ksz = 1, pxsz=1):
    """
    Calculate the HOG correlation values (V statistics) for all unique data pairs in a galaxy's dataset.

    Parameters:
    galaxy (str): Name of the galaxy.
    use_data (str): Specify if you want to use all Pixels or reduced Data.

    Returns:
    pd.DataFrame: DataFrame containing V statistics for unique data pairs.
    """
    
    

    
       
    test_frame1 = marker1
    shapesqrt = int(np.sqrt(np.shape(test_frame1)))
    tf12d = test_frame1.reshape(shapesqrt,-1)

    test_frame2 = marker2
    shapesqrt = int(np.sqrt(np.shape(test_frame2)))
    tf22d = test_frame2.reshape(shapesqrt,-1)
    
    circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_imaLITE(tf12d, tf22d, ksz = ksz, pxsz=pxsz)
        
    
        
    
    
    
    
    return circstats, corrframe, smoothframe1, smoothframe2

def ecHOG_corrframe(marker1 = None, marker2 = None, ksz = 1, pxsz = 1, res = 1):
    """
    Calculate the HOG correlation values (V statistics) for all unique data pairs in a galaxy's dataset.

    Parameters:
    galaxy (str): Name of the galaxy.
    use_data (str): Specify if you want to use all Pixels or reduced Data.

    Returns:
    pd.DataFrame: DataFrame containing V statistics for unique data pairs.
    """
    
    

    
       
    test_frame1 = marker1
    shapesqrt = int(np.sqrt(np.shape(test_frame1)))
    tf12d = test_frame1.reshape(shapesqrt,-1)

    test_frame2 = marker2
    shapesqrt = int(np.sqrt(np.shape(test_frame2)))
    tf22d = test_frame2.reshape(shapesqrt,-1)
    
    circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_frame(tf12d, tf22d, ksz = ksz, pxsz = pxsz, res = res)
        
    rvl = circstats[0]
    Z = circstats[1]
    V = circstats[2]
    VoverVmax = V/np.sum(~np.isnan(corrframe)*(pxsz/ksz)**2)
    circstats={'RVL': rvl, 'Z': Z, 'V': V,'VoverVmax': VoverVmax, "ngood": np.sum(~np.isnan(corrframe))}

    
    
    return circstats, corrframe, smoothframe1, smoothframe2


def ec_HOG_plot_corrframe_galaxy(galaxy = "NGC6060", marker1 = "flux_Halpha_sm", marker2 = "flux_Hbeta_sm", save_img = False):

    gal_dat = pd.read_csv(os.path.join(maindir_path,"EC_HOG/Galaxy_Tables/"+galaxy+"_table.dat"),
                      delim_whitespace=True)

    m1 = marker1
    m2 = marker2

    circstats, corrframe, smoothframe1, smoothframe2 = ecHOG_corrframe(gal_dat[m1].values, gal_dat[m2].values)


    if m1.startswith("flux_AHa"):
        unit_m1 = "mag"
    elif m1.startswith("flux"):
        unit_m1 = r"$10^{16}$ erg / (s $cm^2$)"
    else:
        unit_m1 = "K km / s"

    if m2.startswith("flux_AHa"):
        unit_m2 = "mag"
    elif m2.startswith("flux"):
        unit_m2 = r"$10^{16}$ erg / (s $cm^2$)"
    else:
        unit_m2 = "K km / s"

        
    hist_data = corrframe.ravel() * 180 / np.pi
    hist_data = hist_data[~np.isnan(hist_data)]
    counts, bin_edges = np.histogram(hist_data, bins=45, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_count = np.max(counts)
    half_max = peak_count / 2
    peak_index = np.argmax(counts)
    counts_left = counts[:peak_index + 1]
    bin_centers_left = bin_centers[:peak_index + 1]

    counts_right = counts[peak_index:]
    bin_centers_right = bin_centers[peak_index:]   
    interp_func_left = interp1d(counts_left, bin_centers_left, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_right = interp1d(counts_right, bin_centers_right, kind='linear', bounds_error=False, fill_value="extrapolate")
    left_half_max = interp_func_left(half_max)
    right_half_max = interp_func_right(half_max)
    fwhm = right_half_max - left_half_max
    fwhm = str(np.round(fwhm,2))
    #print(fwhm, type(fwhm))
    fig, axs = plt.subplots(1, 4, figsize=(17, 5),gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    
    cmap1 = axs[0].imshow(smoothframe1, aspect='equal')
    cbar1 = fig.colorbar(cmap1, ax=axs[0], orientation='vertical', aspect=20, shrink=.9)
    axs[0].set_title("Gaussian smoothed marker 1:\n"+clean_marker(m1))
    axs[0].set_xlabel("R.A. pixel")
    axs[0].set_ylabel("Decl. pixel")
    cbar1.set_label(unit_m1, fontsize=12)

    cmap2 = axs[1].imshow(smoothframe2, aspect='equal')
    cbar2 = fig.colorbar(cmap2, ax=axs[1], orientation='vertical', aspect=20, shrink=.9)
    axs[1].set_title("Gaussian smoothed marker 2:\n"+clean_marker(m2))
    axs[1].set_xlabel("R.A. pixel")
    axs[1].set_ylabel("Decl. pixel")
    cbar2.set_label(unit_m2, fontsize=12)

    im = axs[2].imshow(np.abs(corrframe)*180.0/np.pi, cmap="spring", aspect='equal')
    axs[2].set_xlabel("R.A. pixel")
    axs[2].set_ylabel("Decl. pixel")
    cbar3 = fig.colorbar(im, ax=axs[2], orientation='vertical', aspect=20, shrink=.9)
    VoverVmax = circstats["V"]/np.sum(~np.isnan(corrframe))
    axs[2].set_title(r"relative angle $\Phi$," + 
                 r" V/$V_{max}:$ " + str(round(VoverVmax,2)) + 
                 "\navailable pixel: " + str(circstats["ngood"]))
    cbar3.set_label("deg", fontsize=12)
    
    axs[3].hist(hist_data, bins=45, color='blue', alpha=0.7, histtype='step', density=True)
    axs[3].axhline(y=half_max, color='red', linestyle='--', label='Half-Maximum '+fwhm)
    axs[3].axvline(x=left_half_max, color='green', linestyle='--', label='Left FWHM')
    axs[3].axvline(x=right_half_max, color='orange', linestyle='--', label='Right FWHM')
    axs[3].set_xticks(np.arange(-90, 91, 20))
    current_yticks = axs[3].get_yticks()
    axs[3].set_yticks(current_yticks)
    axs[3].set_yticklabels([f'{tick * 100:.2f}' for tick in current_yticks])
    axs[3].set_ylabel("Density / %")
    axs[3].set_xlabel(r"$\Phi$ / deg")
    axs[3].set_title("Density Histogram of Oriented Angles")
    axs[3].legend()
    
    fig.suptitle(f"Galaxy: {galaxy}", fontsize=16, fontweight='bold')

    if save_img:
        plt.savefig("./BA_images/"+galaxy+"_"+m1+"_"+m2+"_corr_Hist_"+str(circstats["ngood"])+".png", dpi = 400)
    
    plt.tight_layout()
    plt.show()

    
    return

def get_V_List(galaxy = "NGC6060", save_list = False):

    unique_pairs = list(combinations(marker, 2))
    gal_data = pd.read_csv("./EC_HOG/Galaxy_Tables/"+galaxy+"_table.dat", delim_whitespace=True)
    
    Vs = []
    VoverVmax = []
    V_used_pixels = []
    for pair in unique_pairs:
        m1, m2 = pair
        circstats, corrframe, smoothframe1, smoothframe2 =ecHOG_corrframe(gal_data[m1].values,gal_data[m2].values)
        Vs.append(circstats["V"])
        VoverVmax.append(circstats["VoverVmax"])
        V_used_pixels.append(np.sum(~np.isnan(corrframe)))
    
    #print(len(Vs),len(VoverVmax),len(V_used_pixels),len(unique_pairs))
    sol_df = pd.DataFrame({
                           "Pair": unique_pairs,
                           "VoverVmax": VoverVmax,
                           "V": Vs,
                           "Vmax": V_used_pixels
                           
                           })
    
    #sol_df = sol_df.sort_values(by="VoverVmax", ascending=False)
    
    if save_list:
        output_filename = f"./EC_HOG/Gal_Vs/{galaxy}_HOG_results.csv"
        sol_df.to_csv(output_filename, index=False)
    
    return

def get_V_List_for_galaxy_list(galaxies):
    failed_galaxies = []

    for galaxy in tqdm(galaxies):
        try:
            get_V_List(galaxy, save_list=True)

        except Exception as e:
            print(f"Failed to process galaxy {galaxy}: {e}")
            failed_galaxies.append(galaxy)

    print(failed_galaxies)
    return failed_galaxies

def get_single_marker(galaxy, marker):


    marker = re.escape(marker)
    gal_dat = pd.read_csv("./EC_HOG/unified_galaxy_Vs.csv", delimiter=",")

    marker_data = gal_dat[gal_dat['Pair'].str.contains(marker, na=False)].reset_index(drop=True)
    #marker_data["corr_part"] = marker_data['Pair'].apply(lambda x: eval(x)[0] if eval(x)[1] == marker else eval(x)[1])
    marker_data = marker_data[['Pair', "Average_VoverVmax", galaxy]]
    

    return marker_data

def all_gal_Vs1():
    dir_path = "./EC_HOG/Gal_Vs/"
    allgal_dic = {}
    galaxy_names = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            galaxy_name = filename.split("_")[0]  
            galaxy_names.append(galaxy_name)
            file_path = os.path.join(dir_path, filename)
            galaxy_data = pd.read_csv(file_path)
            for _, row in galaxy_data.iterrows():
                pair = row["Pair"]
                tuple_value = [row["VoverVmax"], row["V"], row["Vmax"]]
                if pair not in allgal_dic:
                    allgal_dic[pair] = {}
                allgal_dic[pair][galaxy_name] = tuple_value

    pairs = list(allgal_dic.keys())
    columns = ["Pair"] + galaxy_names
    unified_data = []

    for pair in pairs:
        row = [pair]

        VoverVmax_values = [
            allgal_dic[pair][galaxy][0]
            for galaxy in galaxy_names
            if galaxy in allgal_dic[pair]
        ]
        average_VoverVmax = sum(VoverVmax_values) / len(VoverVmax_values) if VoverVmax_values else None
        
        
        row.append(average_VoverVmax)
        
        for galaxy in galaxy_names:
            row.append(allgal_dic[pair].get(galaxy, None))
        unified_data.append(row)
    unified_df = pd.DataFrame(unified_data, columns=columns)


    unified_df.to_csv("./EC_HOG/unified_galaxy_Vs.csv", index=False)
    return

def all_gal_Vs():
    import os
    import pandas as pd

    dir_path = "./EC_HOG/Gal_Vs/"
    allgal_dic = {}
    galaxy_names = []

    
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            galaxy_name = filename.split("_")[0]
            galaxy_names.append(galaxy_name)
            file_path = os.path.join(dir_path, filename)
            galaxy_data = pd.read_csv(file_path)

            for _, row in galaxy_data.iterrows():
                pair = row["Pair"]
                tuple_value = (row["VoverVmax"], row["V"], row["Vmax"])
                if pair not in allgal_dic:
                    allgal_dic[pair] = {}
                allgal_dic[pair][galaxy_name] = tuple_value

    pairs = list(allgal_dic.keys())
    columns = ["Pair", "Average_VoverVmax","std_V_ratio"] + galaxy_names
    unified_data = []

    for pair in pairs:
        row = [pair]
        
        VoverVmax_values = [
            allgal_dic[pair][galaxy][0]
            for galaxy in galaxy_names
            if galaxy in allgal_dic[pair]
        ]
        
        average_VoverVmax = np.nanmean(VoverVmax_values)
        std_VoverVmax = np.nanstd(VoverVmax_values)
        
        row.append(average_VoverVmax)
        row.append(std_VoverVmax)
        
        for galaxy in galaxy_names:
            row.append(allgal_dic[pair].get(galaxy, None))
        
        unified_data.append(row)

    unified_df = pd.DataFrame(unified_data, columns=columns).sort_values("Average_VoverVmax")

    unified_df.to_csv("./EC_HOG/unified_galaxy_Vs_2.csv", index=False)

    return




    

def do_spearmann_matrix(galaxy = "NGC6060", plot_matrix = False):


    spearmann_data = {}
    for m in marker:
        db = get_single_marker(galaxy, m)
        Vratio = [eval(a)[0] for a  in db[galaxy]]
        
        spearmann_data[m] = Vratio
    spearmann_data = pd.DataFrame(spearmann_data)
    spearmann_corr_matrix = spearmann_data.corr(method = "spearman")

    vmin = spearmann_corr_matrix.min().min()  
    vmax = spearmann_corr_matrix.max().max()  
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    mask = np.triu(np.ones_like(spearmann_corr_matrix, dtype=bool))
    #print(spearmann_corr_matrix)
    if plot_matrix == True:
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            spearmann_corr_matrix,
            mask=mask,  
            cmap="coolwarm",  
            norm=norm,
            fmt=".2f",  
            square=True,  
            cbar_kws={"shrink": 0.8}  
        )
        plt.title(f"Spearman Rank Correlation Matrix for {galaxy}", fontsize=16)
        plt.tight_layout()
        plt.show()


    return spearmann_corr_matrix


def do_spider_plot(marker_str="flux_Halpha_sm", galaxy="NGC6060", 
                   plot_matrix = False, save_img = False, plot_img = True, save_path = "../spider_plots/"):
    marker = marker_str
    db = get_single_marker(marker=marker, galaxy=galaxy)

    if marker_str not in np.array([a for pair in db['Pair'] for a in eval(pair)]):
        print(f"marker not in {galaxy}")
        return
    

    categories = np.array([
        a
        for pair in db['Pair']
        for a in eval(pair)
        if a != marker
    ])

    spearman_corr = do_spearmann_matrix_new(galaxy=galaxy, plot_matrix = plot_matrix, save_path=save_path)  

    
    marker_idx = list(spearman_corr.columns).index(marker) 
    correlations = np.abs(spearman_corr.loc[spearman_corr.index[marker_idx], categories].values) 
    

    #values = [eval(a)[0] for a in np.asarray(db[galaxy])]

    values = []
    for a in np.asarray(db[galaxy]):
        try:
            result = eval(a)
            values.append(result[0])
        except:
            values.append(np.nan)
    
    mean_values = db["Average_VoverVmax"]
    
    
    values = np.concatenate((values, [values[0]]))  
    correlations = np.concatenate((correlations, [correlations[0]]))
    mean_values = np.concatenate((mean_values, [mean_values[0]]))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1] 
    argmax_V = np.argmax(values)
    argmax_corr = np.argmax(correlations)
    
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    
    ax.plot(angles, mean_values, color='green', linewidth=2, linestyle='solid', label=r"average $\frac{V}{V_{\text{max}}}$")
    ax.fill(angles, mean_values, color='green', alpha=0.25)
    
    ax.plot(angles, correlations, color='red', linewidth=2, linestyle='solid', label='Spearman Correlations')
    ax.fill(angles, correlations, color='red', alpha=0.25)

    ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid', label=r"$\frac{V}{V_{\text{max}}}$")
    ax.fill(angles, values, color='blue', alpha=0.25)


    ax.scatter(angles[argmax_V], values[argmax_V], color='black', s=50, zorder=5)
    ax.scatter(angles[argmax_corr], correlations[argmax_corr], color='black', s=50, zorder=5)
    ax.scatter(angles[argmax_V], values[argmax_V], color='y', s=30, zorder=5,marker="*")
    ax.scatter(angles[argmax_corr], correlations[argmax_corr], color='y', s=30, zorder=5,marker="*")

    categories = [clean_marker(m) for m in categories] 
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1]) 
    ax.set_xticklabels(categories, fontsize=10)

    #ax.set_title(f"{galaxy}, {marker}", size=14, color='blue', va='bottom')
    #ax.legend(loc='upper right', fontsize=10)
    ax.legend(bbox_to_anchor=(1.3, 1.1), loc='upper right', fontsize=10)
    ax.set_title(f"{galaxy}'s {clean_marker(marker)} correlations", size=16, color='blue', va='top', y=1.1)
    
    if save_img == True:
        plt.savefig(save_path+galaxy+"_"+marker_str+".png", dpi = 400)
    
    if plot_img == True:
        plt.show()

    plt.close()
    return argmax_V == argmax_corr

def do_spider_plot_no_s(marker_str="flux_Halpha_sm", galaxy="NGC6060", plot_matrix = False, save_img = False):
    
    marker = marker_str
    db = get_single_marker(marker=marker, galaxy=galaxy)

    if marker_str not in np.array([a for pair in db['Pair'] for a in eval(pair)]):
        print(f"marker not in {galaxy}")
        return
    

    categories = np.array([
        a
        for pair in db['Pair']
        for a in eval(pair)
        if a != marker
    ])

    spearman_corr = do_spearmann_matrix(galaxy=galaxy, plot_matrix = plot_matrix)  

    
    marker_idx = list(spearman_corr.columns).index(marker) 
    correlations = np.abs(spearman_corr.loc[spearman_corr.index[marker_idx], categories].values) 
    

    #values = [eval(a)[0] for a in np.asarray(db[galaxy])]

    values = []
    for a in np.asarray(db[galaxy]):
        try:
            result = eval(a)
            values.append(result[0])
        except:
            values.append(np.nan)
    
    mean_values = db["Average_VoverVmax"]

    
    values = np.concatenate((values, [values[0]]))  
    correlations = np.concatenate((correlations, [correlations[0]]))
    mean_values = np.concatenate((mean_values, [mean_values[0]]))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1] 

    
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid', label=r"$\frac{V}{V_{\text{max}}}$")
    ax.fill(angles, values, color='blue', alpha=0.25)

    ax.plot(angles, correlations, color='red', linewidth=2, linestyle='solid', label='Spearman Correlations')
    ax.fill(angles, correlations, color='red', alpha=0.25)

    ax.plot(angles, mean_values, color='green', linewidth=2, linestyle='solid', label=r"average $\frac{V}{V_{\text{max}}}$")
    ax.fill(angles, mean_values, color='green', alpha=0.25)


    categories = [clean_marker(m) for m in categories] 
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1]) 
    ax.set_xticklabels(categories, fontsize=10)

    #ax.set_title(f"{galaxy}, {marker}", size=14, color='blue', va='bottom')
    #ax.legend(loc='upper right', fontsize=10)
    ax.legend(bbox_to_anchor=(1.3, 1.1), loc='upper right', fontsize=10)
    ax.set_title(f"{galaxy}'s {marker} correlations", size=16, color='blue', va='top', y=1.1)
    
    if save_img == True:
        plt.savefig("../spider_plots/"+galaxy+"_"+marker_str+"_WRONG.png", dpi = 400)
    plt.show()
    return



def do_spearmann_matrix_new(galaxy = "NGC6060", plot_matrix = False, save_path= "../"):

    spearman_matrix = np.full((len(marker),len(marker)),np.nan)

    galdat = pd.read_csv("./EC_HOG/Galaxy_Tables/"+galaxy+"_table.dat", delim_whitespace=True)
    #print(galdat)
    for i, a in enumerate(marker):
        for j, b in enumerate(marker):
            if i>j:
                
                m1 = galdat[a]
                m2 = galdat[b]
                mask = ~np.isnan(m1) & ~np.isnan(m2)
                m1 = m1[mask]
                m2 = m2[mask]
                corr, _ = spearmanr(m1, m2)
                spearman_matrix[i, j] = corr
                spearman_matrix[j, i] = corr
                spearman_matrix[i, i] = 1

    
    #print(marker_names)
    
    spearmann_corr_matrix = pd.DataFrame(spearman_matrix, index=marker, columns=marker)
    
    marker_names = [clean_marker(m) for m in marker] 
    spearmann_corr_matrix_plot = pd.DataFrame(spearman_matrix, index=marker_names, columns=marker_names)
    


    vmin = spearmann_corr_matrix.min().min()  
    vmax = spearmann_corr_matrix.max().max()  
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    mask_plot = np.triu(np.ones_like(spearmann_corr_matrix, dtype=bool))
    #print(spearmann_corr_matrix)
    if plot_matrix == True:
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            spearmann_corr_matrix,  
            cmap="coolwarm",  
            norm=norm,
            mask=mask_plot,
            fmt=".2f",  
            square=True,  
            cbar_kws={"shrink": 0.8}  
        )
        plt.title(f"Spearman Rank Correlation Matrix for {galaxy}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path+"spearman_matrix_"+galaxy+".png", dpi = 400)
        plt.show()
        

    return spearmann_corr_matrix


def clean_marker(m):
    
    if m == "flux_Halpha_sm":
        m = r"H$_{\alpha}$"
    elif m == "flux_Hbeta_sm":
        m = r"H$_{\beta}$"
    elif m == "flux_AHa_corr_sm":
        m = r"$A_{H_{\alpha}}$"
    elif m.startswith("flux"):
        m = m[5:-3]

    elif m == "mom0_12":
        m = r"I$_{CO}$"
    return m



def ec_HOG_plot_corrframe_galaxy2(galaxy = "NGC6060", marker1 = "flux_Halpha_sm", marker2 = "flux_Hbeta_sm", save_img = False):

    gal_dat = pd.read_csv(os.path.join(maindir_path,"EC_HOG/Galaxy_Tables/"+galaxy+"_table.dat"),
                      delim_whitespace=True)

    m1 = marker1
    m2 = marker2

    circstats, corrframe, smoothframe1, smoothframe2 = ecHOG_corrframe(gal_dat[m1].values, gal_dat[m2].values)


    if m1.startswith("flux_AHa"):
        unit_m1 = "mag"
    elif m1.startswith("flux"):
        unit_m1 = r"$10^{16}$ erg / (s $cm^2$)"
    else:
        unit_m1 = "K km / s"

    if m2.startswith("flux_AHa"):
        unit_m2 = "mag"
    elif m2.startswith("flux"):
        unit_m2 = r"$10^{16}$ erg / (s $cm^2$)"
    else:
        unit_m2 = "K km / s"

        
    hist_data = corrframe.ravel() * 180 / np.pi
    hist_data = hist_data[~np.isnan(hist_data)]
    counts, bin_edges = np.histogram(hist_data, bins=45, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_count = np.max(counts)
    half_max = peak_count / 2
    peak_index = np.argmax(counts)
    counts_left = counts[:peak_index + 1]
    bin_centers_left = bin_centers[:peak_index + 1]

    counts_right = counts[peak_index:]
    bin_centers_right = bin_centers[peak_index:]   
    interp_func_left = interp1d(counts_left, bin_centers_left, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_right = interp1d(counts_right, bin_centers_right, kind='linear', bounds_error=False, fill_value="extrapolate")
    left_half_max = interp_func_left(half_max)
    right_half_max = interp_func_right(half_max)
    fwhm = right_half_max - left_half_max
    fwhm = str(np.round(fwhm,2))
    #print(fwhm, type(fwhm))
    fig, axs = plt.subplots(1, 3, figsize=(13, 5),gridspec_kw={'width_ratios': [1, 1, 1]})
    
    cmap1 = axs[0].imshow(smoothframe1, aspect='equal')
    cbar1 = fig.colorbar(cmap1, ax=axs[0], orientation='vertical', aspect=20, shrink=.9)
    axs[0].set_title("Gaussian smoothed marker 1:\n"+clean_marker(m1), fontsize=12)
    axs[0].set_xlabel("R.A. pixel")
    axs[0].set_ylabel("Decl. pixel")
    cbar1.set_label(unit_m1, fontsize=12)

    cmap2 = axs[1].imshow(smoothframe2, aspect='equal')
    cbar2 = fig.colorbar(cmap2, ax=axs[1], orientation='vertical', aspect=20, shrink=.9)
    axs[1].set_title("Gaussian smoothed marker 2:\n"+clean_marker(m2), fontsize=12)
    axs[1].set_xlabel("R.A. pixel")
    axs[1].set_ylabel("Decl. pixel")
    cbar2.set_label(unit_m2, fontsize=12)

    im = axs[2].imshow(np.abs(corrframe)*180.0/np.pi, cmap="spring", aspect='equal')
    axs[2].set_xlabel("R.A. pixel")
    axs[2].set_ylabel("Decl. pixel")
    cbar3 = fig.colorbar(im, ax=axs[2], orientation='vertical', aspect=20, shrink=.9)
    VoverVmax = circstats["V"]/np.sum(~np.isnan(corrframe))
    axs[2].set_title(r"relative angle $\Phi$," + 
                 r" V/$V_{max}:$ " + str(round(VoverVmax,2)) + 
                 "\navailable pixel: " + str(circstats["ngood"]), fontsize=12)
    cbar3.set_label("deg", fontsize=12)
    
    fig.suptitle(f"Galaxy: {galaxy}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_img:
        plt.savefig("../../Bachelor_images/"+galaxy+"_"+m1+"_"+m2+"_corr_Hist_"+str(circstats["ngood"])+".png", dpi = 400)
    
    
    plt.show()

    
    return