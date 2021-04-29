

# **********************************************************************************************************************
# THESE FUNCTIONS ARE OBSOLETE, BUT FOR NOW I SAVE THEM HERE
# **********************************************************************************************************************
def file2array(filename):
    """
    :param filename: logdir to the .csv file with the array data
    :return: function returns a dictionary with the following information:
        [array]       - key to a NX2 array which holds the scatterers' location in the medium
        [sensitivity] - key to the resulting sensitivity of the array
        [scat_num]    - key to the index of the scatterer which produces the maximal sensitivity
    """
    result_dict = {}
    grid_coords = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
            elif row[4] == '0+0i' and row[5] == '0+0i' and row[6] == '0+0i':
                # last line, extracting the sensitivity
                result_dict['sensitivity'] = float(row[1])
                result_dict['scat_num'] = int(row[0])
            else:  # converting locations to grid coordinates
                x = float(row[0])
                y = float(row[1])
                x_grid = ((x - xRange[0])/xRange[1]) * (xQuantize - 1)
                y_grid = ((y - yRange[0])/yRange[1]) * (yQuantize - 1)
                grid_coords.append(np.array([x_grid, y_grid]))
        result_dict['array'] = np.array(grid_coords)
        return result_dict


def array2mat(arr):
    """
    THIS FILE HOLDS THE FUNCTION WHICH TAKES AN ARRAY OF POINTS AND CONVERTS IT TO A MATRIX, WHERE:
    FOR EACH (X,Y) OF THE MATRIX:
        IF (X,Y) IS IN THE ARRAY, THE INDEX WILL BE 1
        OTHERWISE, IT WILL BE 0
    :param arr: a 2-D array which holds the coordinates of the scatterers
    :return: xQuantize X yQuantize grid simulating the array
    """
    grid_array = torch.ones([1, 1, xQuantize, yQuantize]) * -1
    grid_array[0, 0, arr[:, 1], arr[:, 0]] = 1
    return grid_array


def gather_data(path):
    """
    :param path: holds the logdir to the folder which holds the csv files of the database
    :return:this function goes through all the '.csv' files and extracts the grid points, and the resulting sensitivity.
    the function returns a dictionary with the following keys:
    [scat_arrays] - contains the grid coordinates of the scatterers. Each array hold a set of scatterer locations
    [sensitivity] - contains the matching sensitivity of the respective array of scatterers
    [scat_num]    - contains the number of scatterer that produces the maximal sensitivity for the respective array
    [size]        - hold the size of the database
    """
    # ==================================================================
    # Internal Variables
    # ==================================================================
    file_count = 1
    array_dict = {}
    sensitivity_dict = {}
    scat_num_dict = {}

    # ==================================================================
    # Passing through all the files, and getting the data
    # ==================================================================
    files = os.listdir(path)
    for file in files:
        if file_count > 1000:
            break
        print('processing file number ' + str(file_count))
        fullfile  = path + '\\' + file
        file_dict = file2array(fullfile)

        array_dict[file_count]       = file_dict['array']
        sensitivity_dict[file_count] = file_dict['sensitivity']
        scat_num_dict[file_count]    = file_dict['scat_num']
        file_count += 1

    # ==================================================================
    # Saving and returning the variables
    # ==================================================================
    database = {
        "scat_arrays": array_dict       ,
        "sensitivity": sensitivity_dict ,
        "scat_num"   : scat_num_dict    ,
        "size"        : file_count - 1
    }
    return database


def plot_hist(sensitivity_dict, bins=100, absolute=True):
    """
    :param sensitivity_dict: Dictionary containing the sensitivities in the values
    :param bins: Number of bins in the histogram. default 100
    :param absolute: if True, uses absolute values
    :return: The function plots a histogram of the sensitivities
    """
    sens_vals = list(sensitivity_dict.values())

    if absolute:
        sens_vals = np.abs(sens_vals)

    plt.hist(sens_vals, bins=bins)