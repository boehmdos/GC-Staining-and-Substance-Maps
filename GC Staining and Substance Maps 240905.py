# Import modules
import colour
import colorsys
import math as m
import matplotlib
matplotlib.use("svg")
matplotlib.use("TkAgg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import FreeSimpleGUI as sg
import pandas as pd
import scipy.ndimage as ndimage
import time
from time import sleep
import webbrowser


# This is to get imported files into the exe
def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Load the file
# ======================================================================================================================
sg.theme('DefaultNoMoreNagging')  # Add some colour to the window interface

# Create all layouts for GUI
layout1 = [
    [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(file_types=(("CommaSeparatedValues", "*.csv"),))],
    [sg.Text("Should the first line of the file be ignored (headers, not data)?"), sg.Checkbox("Yes")],
    [sg.Text("Which delimiter is used in the csv file?"), sg.Input(',', size=(8, 20))],
    [sg.Text("Select m/z range within [29,200]:"), sg.Checkbox("Auto"), sg.Text("From:"), sg.Input(size=(8, 20)), sg.Text("To:"), sg.Input(size=(8, 20))],
    [sg.Text("Select time section [min]:"), sg.Checkbox("All"), sg.Text("From:"), sg.Input(size=(8, 20)), sg.Text("To:"), sg.Input(size=(8, 20))],
    [sg.Text("Minimum TIC intensity:"), sg.Input('100000', size=(8, 20))],
    [sg.Text("Exponent to amplify smaller peaks (0<a=<1): "), sg.Text("a="), sg.Input('1', size=(8, 20))],
    [sg.Text("Choose SOM file: "), sg.Input(key='SOM'), sg.FileBrowse(file_types=(("Text Files", "*.txt"),))],
    [sg.Text("Input SOM dimensions:"), sg.Text("x:"), sg.Input('256', size=(8, 20)), sg.Text("y:"), sg.Input('256', size=(8, 20))],
    [sg.Text("Choose a folder to save output: "), sg.Input(), sg.FolderBrowse()],
    [sg.Text("(Optional) Select output file format for the figures to be saved directly:"), sg.Checkbox(".png"), sg.Checkbox(".svg (only for chromatogram)")],
    [sg.Button("Submit"), sg.Button("Cancel"), sg.Button("Help")]
]

layout2 = [
    [sg.Text("Your data have been processed. The output has been saved in the pre-defined folder.")],
    [sg.Text("Do you want to stain only a certain area of the map? "), sg.Button("Yes"), sg.Button("No")]
]

layout3 = [
    [sg.Text("Define new center:"), sg.Text("x:"), sg.Input('127.5', size=(5, 15)), sg.Text("y:"), sg.Input('127.5', size=(5, 15))],
    [sg.Text("Define new radius from center (r<180):"), sg.Text("r:"), sg.Input('180.3', size=(5, 15))],
    [sg.Text("Define rotation of the color plane in degrees:"), sg.Input('0', size=(5, 15))],
    [sg.Text("Exponent to amplify smaller peaks (0<a=<1): "), sg.Text("a="), sg.Input('1', size=(8, 20))],
    [sg.Text("(Optional) Select output files format for the figures to be saved directly:"), sg.Checkbox(".png"), sg.Checkbox(".svg (only for chromatogram)")],
    [sg.Button("Check position"), sg.Button("Submit and stain"), sg.Button("Cancel")]
]

layout_progressbar1 = [
    [sg.Text("Your data is being processed...")],
    [sg.ProgressBar(max_value=10, orientation='h', size=(20, 20), key='progress')]
]

layout_progressbar2 = [
    [sg.Text("The staining recipe is being made...")],
    [sg.ProgressBar(max_value=10, orientation='h', size=(20, 20), key='progress_stain')]
]

# Create the Window
window = sg.Window('File browsing', layout1, font=("Helvetica", 12))
# Create event loop
valid = False
while not valid:
    event, values = window.read()
    print(values)
    # End program if user closes window or presses "Cancel" button
    if event == "Cancel" or event == sg.WINDOW_CLOSED:
        break
    # Redirect to help file
    elif event == "Help":
        url = resource_path("Howto.html")
        valid = False
        webbrowser.open_new(url)
    # Export data
    elif event == "Submit":
        window.close()
        # Add waiting window
        window = sg.Window('Progress Meter', layout_progressbar1,
                           finalize=True, keep_on_top=True, font=("Helvetica", 12))
        # Get the element to make updating easier
        progress_bar = window['progress']

        start = time.time()  # Start counting the runtime

        # Read the csv file
        if values[1]:
            data = pd.read_csv(values[0], header=None, delimiter=values[2], low_memory=False).to_numpy()
        else:
            data = pd.read_csv(values[0], delimiter=values[2], low_memory=False).to_numpy()

        # Setting up for mz range selection
        fields = np.int16(data[0, 3:])

        if True:
            mz_min = np.int16(fields.min())
            mz_max = np.int16(fields.max())

            # Clipping data to be in range of m/z = [29,200]
            if mz_min == 29:
                mz_min_loc = 3
            elif mz_min < 29:
                mz_min = 29
                mz_min_loc = np.where(fields == 29)
                mz_min_loc = int(mz_min_loc[0]) + 3
            else:
                mz_min_loc = np.where(fields == mz_min)
                mz_min_loc = int(mz_min_loc[0]) + 3

            if mz_max == 200:
                mz_max_loc = np.where(fields == 200)
                mz_max_loc = int(mz_max_loc[0]) + 4
            elif mz_max > 200:
                mz_max = 200
                mz_max_loc = np.where(fields == 200)
                mz_max_loc = int(mz_max_loc[0]) + 4  # +4 is to counter the clip of the places 0,1,2 in the initial data array
            else:
                mz_max_loc = np.where(fields == mz_max)
                mz_max_loc = int(mz_max_loc[0]) + 4

        # User's desired mz range
        else:
            if values[4] == '':
                mz_min = np.int16(fields.min())
                if mz_min < 29:
                    mz_min = 29
                    mz_min_loc = np.where(fields == 29)
                    mz_min_loc = int(mz_min_loc[0]) + 3
                else:
                    mz_min_loc = np.where(fields == mz_min)
                    mz_min_loc = int(mz_min_loc[0]) + 3
            elif int(values[4]) < 29:
                sg.popup_error("Wrong m/z range selection")
                break
            else:
                mz_min_loc = np.where(fields == (values[4]))
                mz_min_loc = int(mz_min_loc[0]) + 3
            if values[5] == '':
                mz_max = np.int16(fields.max())
                if mz_max > 200:
                    mz_max = 200
                    mz_max_loc = np.where(fields == 200)
                    mz_max_loc = int(mz_max_loc[0]) + 4
                else:
                    mz_max_loc = np.where(fields == mz_max)
                    mz_max_loc = int(mz_max_loc[0]) + 4
            elif int(values[4]) < 29 or int(values[5]) > 200:
                sg.popup_error("Wrong m/z range selection")
                break
            else:
                mz_max_loc = np.where(fields == float(values[5]))
                mz_max_loc = int(mz_max_loc[0]) + 4

        # Final range of m/z
        fields = np.int16(data[0, mz_min_loc:mz_max_loc])

        # Spot #1 to indicate progress
        progress_bar.update_bar(1)  # show 10% complete
        sleep(2)

        # Setting up for time section
        RT = np.float32(data[1:, :2])

        if values[6]:
            RT_min_idx = 1
            RT_max_idx = (np.shape(RT)[0])
        else:
            if values[7] == '':
                values[7] = 0
                RT_min_idx = 1
            else:
                RT_min_idx = (np.abs(RT[:, 1] - float(values[7]))).argmin()
            if values[8] == '':
                values[8] = m.ceil(np.max(RT[1]))
                RT_max_idx = (np.shape(RT)[0])
            else:
                RT_max_idx = (np.abs(RT[:, 1] - float(values[8]))).argmin()
        RT = RT[RT_min_idx:RT_max_idx, :]

        # Reshape all data to match field size and time section
        data = data[RT_min_idx:RT_max_idx, mz_min_loc - 3:mz_max_loc]

        # Normalize MS values
        mz_data = np.float32(data[:, 3:])  # Have a 2d-array of only floats
        intensity = mz_data  # Need to save somewhere the initial values of ions to find later TIC -> Luminance
        mz_data = mz_data / np.amax(mz_data, axis=1)[:, None]

        # Spot #2 to indicate progress
        progress_bar.update_bar(2)  # show 20% complete
        sleep(2)

        # Load SOM. This SOM is 256x256x172 (xyz) or (yxz) doesn't matter.
        SOM = pd.read_csv(values['SOM'], header=None, low_memory=False).to_numpy().reshape((-1,))
        x, y = int(values[11]), int(values[12])

        # Getting the z-vectors in the form of matrix.
        z_matrix = np.ndarray(shape=(x*y, np.shape(SOM)[0]//x//y), dtype='float32')
        for j in range(x*y):
            z_matrix[j, :] = SOM[j::x*y]

        # Make z_matrix vectors in the same m/z range as the data.
        if np.shape(z_matrix)[1] != np.shape(mz_data)[1]:
            z_matrix = z_matrix[:, fields[0]-29:fields[-1]+fields[0]-29+1]

        # Spot #3 to indicate progress
        progress_bar.update_bar(4)  # show 40% complete
        sleep(2)

        # Find BMU for each MS (row of mz_data)
        bmu_idx = []
        D = []  # stores difference between MS and weight
        for j in range(np.shape(mz_data)[0]):
            difference = np.sum((abs(mz_data[j] - z_matrix)), axis=1)
            bmu_idx.append(np.argmin(difference))
            D.append(np.min(difference))

        D = np.array(D)

        # Spot #4 to indicate progress
        progress_bar.update_bar(6)  # show 60% complete
        sleep(2)

        # Determine colour according to BMU coordinates
        bmu_idx = np.array(bmu_idx)
        center = x / 2 - 0.5, y / 2 - 0.5
        xy = np.array([bmu_idx % x, bmu_idx // y])

        # Get the radius -> Saturation
        r_max = ((x / 2 - 0.5) ** 2 + (y / 2 - 0.5) ** 2) ** 0.5  # Saturation is 1
        stain = np.array([xy[0]-center[0], xy[1]-center[1]])

        length = []
        for i in range(np.shape(stain)[1]):
            length.append(np.linalg.norm(stain[:, i]) / r_max)
        S = np.array(length)

        # Get the phase -> Hue
        unit_stain = np.ndarray(shape=np.shape(stain))
        for i in range(np.shape(stain)[1]):
            unit_stain[:, i] = stain[:, i] / np.linalg.norm(stain[:, i])
        unit_north = np.array([0, 1])
        # Divide into sections for x<127.5 and x>127.5
        phase = []  # in radians
        for i in range(np.shape(stain)[1]):
            if xy[0, i] < x/2 - 0.5:
                phase.append(2 * np.pi - np.arccos(np.dot(unit_stain[:, i], unit_north)))
            else:
                phase.append(np.arccos(np.dot(unit_stain[:, i], unit_north)))
        phase = np.array(phase)
        H = phase / (2*np.pi)

        # Get the signal intensity -> Luminance [0.5,1]
        # Adding noise filter
        if values[9] == '':
            noise_limit = 0
        else:
            noise_limit = float(values[9])
        TIC = intensity.sum(axis=1)
        TIC_plot = intensity.sum(axis=1)
        TIC[TIC < noise_limit] = 0
        # Checking correct value of exponent
        if values[10] == '':
            a = 1
        else:
            a = float(values[10])
        if a < 0 or a > 1:
            sg.popup_error("Wrong exponent")
            break
        L = 1.0 - 0.5 * (TIC / np.amax(TIC, axis=0)) ** a

        HSL = np.column_stack((H, S, L))

        # Convert HSL to RGB
        color = []
        for i in range(np.shape(H)[0]):
            color.append(colorsys.hls_to_rgb(H[i], L[i], S[i]))
        color = np.array(color)
        color = color.reshape(np.shape(color)[0], -1)

        # Determine BMU intensity
        bmu_unique = np.unique(bmu_idx)
        xy_unique = np.array([bmu_unique % 256, bmu_unique // 256])

        TIC_bmu = []  # sum of TIC for same BMUs
        for i in range(len(bmu_unique)):
            idx = (np.where(bmu_idx == bmu_unique[i])[0])
            TIC_bmu.append(TIC[idx].sum())

        TIC_bmu = np.array(TIC_bmu)
        TIC_bmu_ln = np.log(TIC_bmu, out=TIC_bmu, where=(TIC_bmu>=1))

        # Color gradient according to BMU intensity
        black = colour.Color("black")
        white = colour.Color("white")
        G = 1 - TIC_bmu / np.amax(TIC_bmu) #0 is black, 1 is white
        ones = np.ones_like(G)
        gradient = (np.column_stack((G, G, G, ones)))

        # Spot #5 to indicate progress
        progress_bar.update_bar(8)  # show 80% complete
        sleep(2)

        # Get saving path
        path = values.get(13)

        # Data for the output file
        input_name = str(values.get(0)).rpartition('/')[-1]
        timestr = time.strftime("%Y%m%d_%H%M%S")
        # SOM_name = str(values.get(SOM)).rpartition('/')[-1]
        if values[6]:
            time_range = str('All')
        else:
            time_range = '[' + str(values[7]) + '-' + str(values[8]) + ']'
        mz_range = '[' + str(fields[0]) + '-' + str(fields[-1]) + ']'

        parameters_1 = np.array([input_name, timestr, time_range, mz_range, noise_limit, a])[np.newaxis]
        parameters_2 = np.column_stack((TIC_plot, TIC, D, xy.T))
        exported = np.concatenate((RT, parameters_2, HSL, 255*color), axis=1)
        parameters_BMU = np.column_stack((bmu_unique, TIC_bmu, TIC_bmu_ln, G))

        df1 = pd.DataFrame(parameters_1,
                           columns=['Input File Name', 'Date Created','Time Range[min]', 'm/z Range',
                                    'Noise Limit[cts]', 'exponent'])
        df2 = pd.DataFrame(exported,
                           columns=['RT(ms)', 'RT(min)', 'TIC[cts]', 'TIC[cts] denoised', 'Difference', 'x', 'y', 'H',
                                    'S', 'L', 'R', 'G', 'B'])
        df3 = pd.DataFrame(parameters_BMU, columns=['unique BMU', 'TIC', 'ln(TIC)', 'Grayscale Value'])

        path_outfile = path + '/results_' + timestr + '_' + input_name
        path_BMU_fingerprint = path + '/BMU_fingerprint_' + timestr + '_' + input_name

        # Spot #6 to indicate progress
        progress_bar.update_bar(9)  # show 90% complete
        sleep(2)

        # Plot
        RT_min = RT[:, 1]
        color = matplotlib.colors.to_rgba_array(color)

        # Chromatogram
        plt.figure(1, figsize=(6, 8))
        plt.xlabel("Intensity [cts]")
        plt.ylabel("Retention Time [min]")
        plt.plot(TIC_plot, RT_min, c='black', lw=0.8)
        plt.gca().invert_yaxis()
        plt.grid(False)
        for i in range(np.shape(color)[0]-1):
            plt.fill_between(TIC_plot, RT_min[i], RT_min[i+1], color=(color[i, :]))
        plt.tight_layout()
        # Optional save
        if values[14]:
            path_chromatogram = path + '/chromatogram_' + str(a) + '.png'
            plt.savefig(path_chromatogram)
        if values[15]:
            path_chromatogram = path + '/chromatogram_' + str(a) + '.svg'
            plt.savefig(path_chromatogram)
        plt.pause(1)

        # BMU map
        img = mpimg.imread(resource_path('FullColorWheel.png'))
        x0_wheel_l = - (2 * r_max - x) / 2
        x0_wheel_u = x + (2 * r_max - x) / 2
        y0_wheel_l = - (2 * r_max - y) / 2
        y0_wheel_u = y + (2 * r_max - y) / 2

        cm = plt.cm.ScalarMappable(cmap='Greys')
        cm.set_array([min(TIC_bmu), max(TIC_bmu)])

        plt.figure(2, figsize=(x*4*0.010416667, y*4*0.010416667))  # Convert (pixel to in) * 4 to have a nice ratio
        plt.title("BMU Map")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(xy_unique[0], xy_unique[1], color=gradient, marker='.')
        cbar = plt.colorbar(cm, location='right', ticklocation='right', pad=0.025, aspect=40)
        cbar.set_label(label='ln(TIC) for each BMU [cts]', rotation=270, labelpad=12)
        cbar.minorticks_on()
        plt.imshow(img, extent=[x0_wheel_l-1, x0_wheel_u+1, y0_wheel_l-1, y0_wheel_u+1], aspect='auto')
        plt.grid(True, which='both', color='black', lw=0.2)
        plt.xlim(0, x)
        plt.ylim(0, y)
        plt.tight_layout()
        # Optional save
        if values[14]:
            path_bmumap = path + '/BMU_map.png'
            plt.savefig(path_bmumap)
        plt.pause(1)

        # Spot #7 to indicate progress
        progress_bar.update_bar(10)  # show 100% complete
        sleep(2)
        window.close()

        end = time.time()  # Basic runtime completed
        print(end - start)

        # Staining recipes
        # ==============================================================================================================

        # Create window for staining recipes
        window = sg.Window('Staining recipes', layout2, font=("Helvetica", 12))

        # Create event loop
        valid_stain = True
        while valid_stain:
            event, values = window.read()
            # End program if user selects 'No' or closes window
            if event == "No" or event == sg.WINDOW_CLOSED:
                window.close()

                # Saving analyzed data
                df1.to_csv(path_outfile, index=False, mode='w')
                df2.to_csv(path_outfile, sep=',', index=False, mode='a')
                df3.to_csv(path_BMU_fingerprint, index=False, mode='w')

                # Exit loop
                valid_stain = False
            # Proceed
            elif event == "Yes":
                window.close()
                # Create window to input new parameters
                window = sg.Window('New parameters', layout3, keep_on_top=True, font=("Helvetica", 12))

                valid_input = False
                # Create event loop
                while not valid_input:
                    event, values = window.read()
                    # Exit program if user closes window or presses "Cancel" button
                    if event == 'Cancel' or event == sg.WINDOW_CLOSED:
                        window.close()
                    # Preview
                    elif event == 'Check position':
                        # Clear figure to redraw
                        plt.figure(3).clear()

                        x0 = float(values[0])
                        y0 = float(values[1])
                        r = float(values[2])

                        # Set limits for plane and prepare BMU Map axis
                        x0_lower, x0_upper = m.floor(x0 - r), m.ceil(x0 + r)
                        y0_lower, y0_upper = m.floor(y0 - r), m.ceil(y0 + r)

                        # Save (x,y)(low,upper) values for color wheel
                        x0_wheel_l = x0_lower
                        x0_wheel_u = x0_upper
                        y0_wheel_l = y0_lower
                        y0_wheel_u = y0_upper

                        if x0_lower <= 0:
                            x0_lower = 0
                        if x0_upper >= x:
                            x0_wheel_u = x0_upper
                            x0_upper = x
                        if y0_lower <= 0:
                            y0_wheel_l = y0_lower
                            y0_lower = 0
                        if y0_upper >= y:
                            y0_wheel_u = y0_upper
                            y0_upper = y

                        x_range = np.arange(x0_lower, x0_upper + 1, 1.)
                        y_range = np.arange(y0_lower, y0_upper + 1, 1.)

                        # Spot #1 to indicate progress
                        progress_bar.update_bar(1)  # show 10% complete
                        sleep(2)

                        xy_user = []
                        idx_in = []  # A list of all the indices inside staining area
                        idx_out = []  # A list of all the indices outside staining area
                        # Keep only x,y coordinates within limits
                        for i in range(np.shape(xy)[1]):
                            if (xy[0, i] >= x_range[0]) and (xy[0, i] <= x_range[-1]) \
                                    and (xy[1, i] >= y_range[0]) and (xy[1, i] <= y_range[-1]):
                                xy_user.append(xy[0, i])
                                xy_user.append(xy[1, i])
                                idx_in.append(i)
                            else:
                                idx_out.append(i)
                        xy_user = np.array([xy_user[::2], xy_user[1::2]])

                        # Spot #2 to indicate progress
                        progress_bar.update_bar(4)  # show 40% complete
                        sleep(2)

                        # Find new color recipe
                        center_stain = x0, y0
                        S_stain, H_stain = np.zeros(shape=np.shape(S)), np.zeros(shape=np.shape(H))
                        stain_user = np.array([xy_user[0] - center_stain[0], xy_user[1] - center_stain[1]])

                        # Get Saturation
                        length_user = []
                        idx_stain = []
                        for i in range(np.shape(stain_user)[1]):
                            # Get points outside circled area
                            if (np.linalg.norm(stain_user[:, i]) / r) > 1:
                                idx_stain.append(i)
                                length_user.append(0)
                            else:
                                length_user.append(np.linalg.norm(stain_user[:, i]) / r)
                        np.put(S_stain, idx_in, length_user)

                        # Get Hue
                        unit_stain_user = np.ndarray(shape=np.shape(stain_user))
                        for i in range(np.shape(stain_user)[1]):
                            unit_stain_user[:, i] = stain_user[:, i] / np.linalg.norm(stain_user[:, i])
                        # Divide into sections for x<xcenter and x>xcenter
                        phase_user = []  # in radians
                        for i in range(np.shape(stain_user)[1]):
                            if xy_user[0, i] < (center_stain[0] - 0.5):
                                phase_user.append(2 * np.pi - np.arccos(np.dot(unit_stain_user[:, i], unit_north)))
                            else:
                                phase_user.append(np.arccos(np.dot(unit_stain_user[:, i], unit_north)))
                        phase_user = np.array(phase_user)
                        # Add initial phase of user's input
                        img = mpimg.imread(resource_path('FullColorWheel.png'))
                        phi = float(values[3])
                        if phi >= 360:
                            phi = phi % 360
                        img = ndimage.rotate(img, phi, reshape=False)
                        phase_user = (phase_user + (phi * np.pi / 180)) / (2 * np.pi)
                        np.put(H_stain, idx_in, phase_user)

                        # Get the luminance
                        if values[4] == '':
                            b = 1
                        else:
                            b = float(values[4])
                        L_stain = 1.0 - 0.5 * (TIC / np.amax(TIC, axis=0)) ** b
                        # Ignore for BMUs outside plane of interest, making them appear white
                        np.put(L_stain, idx_out, 1.0)
                        # Ignore for BMUs outside circled area, making them appear white
                        np.put(L_stain, idx_stain, 1.0)

                        HSL_stain = np.column_stack((H_stain, S_stain, L_stain))

                        # Convert HSL to RGB
                        color_user = []
                        for i in range(np.shape(H_stain)[0]):
                            color_user.append(colorsys.hls_to_rgb(H_stain[i], L_stain[i], S_stain[i]))
                        color_user = np.array(color_user)
                        color_user = color_user.reshape(np.shape(color_user)[0], -1)

                        # Spot #3 to indicate progress
                        progress_bar.update_bar(8)  # show 80% complete
                        sleep(2)

                        # Save to the output file
                        center_stain = '[' + str(x0) + ',' + str(y0) + ']'
                        parameters_3 = np.array([input_name, timestr, time_range, mz_range, noise_limit, a,
                                                 center_stain, r, phi, b])[np.newaxis]
                        exported_stain = np.concatenate((exported, HSL_stain, 255 * color_user), axis=1)

                        df1_1 = pd.DataFrame(parameters_3,
                                             columns=['Input File Name', 'Date Created', 'Time Range[min]', 'm/z Range',
                                                      'Noise Limit[cts]', 'exponent', 'Stain Center', 'Radius',
                                                      'Rotation[degrees]', 'exponent (stain)'])
                        df2_1 = pd.DataFrame(exported_stain,
                                             columns=['RT[ms]', 'RT[min]', 'TIC[cts]', 'TIC[cts] de-noised', 'distance',
                                                      'x', 'y', 'H', 'S', 'L', 'R', 'G', 'B', 'H (stain)', 'S (stain)',
                                                      'L (stain)', 'R (stain)', 'G (stain)', 'B (stain)'])

                        df1_1.to_csv(path_outfile, index=False, mode='w')
                        df2_1.to_csv(path_outfile, sep=',', index=False, mode='a')

                        # Plot new graphs of chromatogram and BMU map
                        color_user = matplotlib.colors.to_rgba_array(color_user)

                        # Fix projecting color plane
                        img = mpimg.imread(resource_path('FullColorWheel.png'))
                        phi = float(values[3])
                        if phi >= 360:
                            phi = phi % 360
                        img = ndimage.rotate(img, phi, reshape=False)

                        # New BMU map
                        plt.figure(3, figsize=(x * 4 * 0.010416667, y * 4 * 0.010416667))
                        plt.title("BMU Map Staining Recipe")
                        plt.xlabel("x")
                        plt.ylabel("y")
                        plt.imshow((img * 255).astype(np.uint8),
                                   extent=[x0_wheel_l, x0_wheel_u, y0_wheel_l, y0_wheel_u], aspect='auto')
                        plt.scatter(xy_unique[0], xy_unique[1], color=gradient, marker='.')
                        cbar = plt.colorbar(cm, location='right', ticklocation='right', pad=0.025, aspect=40)
                        cbar.set_label(label='ln(TIC) for each BMU [cts]', rotation=270, labelpad=12)
                        cbar.minorticks_on()
                        plt.gcf().gca().add_artist(plt.Circle((x0, y0), r, fill=False))
                        plt.xlim(0, x)
                        plt.ylim(0, y)
                        plt.grid(True, which='both', color='black', lw=0.1)
                        plt.tight_layout()
                        plt.show(block=False)

                        valid_input = False
                    # Illustrate new map according to inputs
                    elif event == 'Submit and stain':
                        window.close()
                        # Add waiting window
                        window = sg.Window('Progress Meter', layout_progressbar2,
                                           finalize=True, keep_on_top=True, font=("Helvetica", 12))
                        # Get the element to make updating easier
                        progress_bar = window['progress_stain']

                        x0 = float(values[0])
                        y0 = float(values[1])
                        r = float(values[2])

                        # Set limits for plane and prepare BMU Map axis
                        x0_lower, x0_upper = m.floor(x0 - r), m.ceil(x0 + r)
                        y0_lower, y0_upper = m.floor(y0 - r), m.ceil(y0 + r)

                        # Save (x,y)(low,upper) values for color wheel
                        x0_wheel_l = x0_lower
                        x0_wheel_u = x0_upper
                        y0_wheel_l = y0_lower
                        y0_wheel_u = y0_upper

                        if x0_lower <= 0:
                            x0_lower = 0
                        if x0_upper >= x:
                            x0_wheel_u = x0_upper
                            x0_upper = x
                        if y0_lower <= 0:
                            y0_wheel_l = y0_lower
                            y0_lower = 0
                        if y0_upper >= y:
                            y0_wheel_u = y0_upper
                            y0_upper = y

                        x_range = np.arange(x0_lower, x0_upper + 1, 1.)
                        y_range = np.arange(y0_lower, y0_upper + 1, 1.)

                        # Spot #1 to indicate progress
                        progress_bar.update_bar(1)  # show 10% complete
                        sleep(2)

                        xy_user = []
                        idx_in = []  # A list of all the indices inside staining area
                        idx_out = []  # A list of all the indices outside staining area
                        # Keep only x,y coordinates within limits
                        for i in range(np.shape(xy)[1]):
                            if (xy[0, i] >= x_range[0]) and (xy[0, i] <= x_range[-1]) \
                                    and (xy[1, i] >= y_range[0]) and (xy[1, i] <= y_range[-1]):
                                xy_user.append(xy[0, i])
                                xy_user.append(xy[1, i])
                                idx_in.append(i)
                            else:
                                idx_out.append(i)
                        xy_user = np.array([xy_user[::2], xy_user[1::2]])

                        # Spot #2 to indicate progress
                        progress_bar.update_bar(4)  # show 40% complete
                        sleep(2)

                        # Find new color recipe
                        center_stain = x0, y0
                        S_stain, H_stain = np.zeros(shape=np.shape(S)), np.zeros(shape=np.shape(H))
                        stain_user = np.array([xy_user[0] - center_stain[0], xy_user[1] - center_stain[1]])

                        # Get Saturation
                        length_user = []
                        idx_stain = []
                        for i in range(np.shape(stain_user)[1]):
                            # Get points outside circled area
                            if (np.linalg.norm(stain_user[:, i]) / r) > 1:
                                idx_stain.append(i)
                                length_user.append(0)
                            else:
                                length_user.append(np.linalg.norm(stain_user[:, i]) / r)
                        np.put(S_stain, idx_in, length_user)

                        # Get Hue
                        unit_stain_user = np.ndarray(shape=np.shape(stain_user))
                        for i in range(np.shape(stain_user)[1]):
                            unit_stain_user[:, i] = stain_user[:, i] / np.linalg.norm(stain_user[:, i])
                        # Divide into sections for x<xcenter and x>xcenter
                        phase_user = []  # in radians
                        for i in range(np.shape(stain_user)[1]):
                            if xy_user[0, i] < (center_stain[0] - 0.5):
                                phase_user.append(2 * np.pi - np.arccos(np.dot(unit_stain_user[:, i], unit_north)))
                            else:
                                phase_user.append(np.arccos(np.dot(unit_stain_user[:, i], unit_north)))
                        phase_user = np.array(phase_user)
                        # Add initial phase of user's input
                        img = mpimg.imread(resource_path('FullColorWheel.png'))
                        phi = float(values[3])
                        if phi >= 360:
                            phi = phi % 360
                        img = ndimage.rotate(img, phi, reshape=False)
                        phase_user = (phase_user + (phi * np.pi / 180)) / (2 * np.pi)
                        np.put(H_stain, idx_in, phase_user)

                        # Get the luminance
                        if values[4] == '':
                            b = 1
                        else:
                            b = float(values[4])
                        L_stain = 1.0 - 0.5 * (TIC / np.amax(TIC, axis=0)) ** b
                        # Ignore for BMUs outside plane of interest, making them appear white
                        np.put(L_stain, idx_out, 1.0)
                        # Ignore for BMUs outside circled area, making them appear white
                        np.put(L_stain, idx_stain, 1.0)

                        HSL_stain = np.column_stack((H_stain, S_stain, L_stain))

                        # Convert HSL to RGB
                        color_user = []
                        for i in range(np.shape(H_stain)[0]):
                            color_user.append(colorsys.hls_to_rgb(H_stain[i], L_stain[i], S_stain[i]))
                        color_user = np.array(color_user)
                        color_user = color_user.reshape(np.shape(color_user)[0], -1)

                        # Spot #3 to indicate progress
                        progress_bar.update_bar(8)  # show 80% complete
                        sleep(2)

                        # Save to the output file
                        center_stain = '[' + str(x0) + ',' + str(y0) + ']'
                        parameters_3 = np.array([input_name, timestr, time_range, mz_range, noise_limit, a,
                                                 center_stain, r, phi, b])[np.newaxis]
                        exported_stain = np.concatenate((exported, HSL_stain, 255 * color_user), axis=1)

                        df1_1 = pd.DataFrame(parameters_3,
                                             columns=['Input File Name', 'Date Created', 'Time Range[min]', 'm/z Range',
                                                      'Noise Limit[cts]', 'exponent', 'Stain Center', 'Radius',
                                                      'Rotation[degrees]', 'exponent (stain)'])
                        df2_1 = pd.DataFrame(exported_stain,
                                             columns=['RT[ms]', 'RT[min]', 'TIC[cts]', 'TIC[cts] de-noised', 'distance',
                                                      'x', 'y', 'H', 'S', 'L', 'R', 'G', 'B', 'H (stain)', 'S (stain)',
                                                      'L (stain)', 'R (stain)', 'G (stain)', 'B (stain)'])

                        df1_1.to_csv(path_outfile, index=False, mode='w')
                        df2_1.to_csv(path_outfile, sep=',', index=False, mode='a')

                        # Plot new graphs of chromatogram and BMU map
                        color_user = matplotlib.colors.to_rgba_array(color_user)

                        # Chromatogram for staining recipe
                        plt.figure(4, figsize=(6, 8))
                        plt.xlabel("Intensity [cts]")
                        plt.ylabel("Retention Time [min]")
                        plt.plot(TIC_plot, RT_min, c='black', lw=0.8)
                        plt.gca().invert_yaxis()
                        plt.grid(False)
                        for i in range(np.shape(color_user)[0] - 1):
                            plt.fill_between(TIC_plot, RT_min[i], RT_min[i + 1], color=(color_user[i, :]))
                        plt.tight_layout()
                        if values[5]:
                            path_chromatogram_stain = path + '/chromatogram_' + str(a) + '_recipe' + str(b) + '.png'
                            plt.savefig(path_chromatogram_stain)
                        if values[6]:
                            path_chromatogram_stain = path + '/chromatogram_' + str(a) + '_recipe' + str(b) + '.svg'
                            plt.savefig(path_chromatogram_stain)

                        # BMU map for staining recipe
                        plt.figure(5, figsize=(x*4*0.010416667, y*4*0.010416667))
                        plt.title("BMU Map Staining Recipe")
                        plt.xlabel("x")
                        plt.ylabel("y")
                        plt.imshow((img * 255).astype(np.uint8),
                                   extent=[x0_wheel_l, x0_wheel_u, y0_wheel_l, y0_wheel_u], aspect='auto')
                        plt.scatter(xy_unique[0], xy_unique[1], color=gradient, marker='.')
                        cbar = plt.colorbar(cm, location='right', ticklocation='right', pad=0.025, aspect=40)
                        cbar.set_label(label='ln(TIC) for each BMU [cts]', rotation=270, labelpad=12)
                        cbar.minorticks_on()
                        plt.gcf().gca().add_artist(plt.Circle((x0, y0), r, fill=False))
                        plt.xlim(0, x)
                        plt.ylim(0, y)
                        plt.grid(True, which='both', color='black', lw=0.1)
                        plt.tight_layout()
                        # Optional save
                        if values[5]:
                            path_bmumap_stain = \
                                path + '/BMU_map_staining_area_' + str(x0) + '_' + str(y0) + '_' + str(r) + '.png'
                            plt.savefig(path_bmumap_stain)

                        # Spot #4 to indicate progress
                        progress_bar.update_bar(10)  # show 10% complete
                        sleep(2)
                        window.close()

                        plt.show()
                        break

    # Keep on showing plots if user doesn't select to continue with staining recipes
    plt.show()
    break
