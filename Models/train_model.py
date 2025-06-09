from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# define experimental training data set
TRAIN_DATA = [
    "fft data/Composite/new_2_lamps_left_off_fft.npy",
    "fft data/Composite/new_2_lamps_right_off_fft.npy",
    "fft data/Composite/new_2_lamps_top_off_fft.npy",
    "fft data/Composite/new_3_lamps_fft.npy",
    "fft data/Resin/circular_1_lamp_left_on_fft.npy",
    "fft data/Resin/circular_1_lamp_right_on_fft.npy",
    "fft data/Resin/circular_1_lamp_top_on_fft.npy",
    "fft data/Resin/circular_2_lamps_left_off_fft.npy",
    "fft data/Resin/circular_3_lamps_fft.npy",
    "fft data/Resin/rec_1_lamp_left_on_fft.npy",
    "fft data/Resin/rec_1_lamp_right_on_fft.npy",
    "fft data/Resin/rec_2_lamps_left_off_fft.npy",
    "fft data/Resin/rec_2_lamps_right_off_fft.npy",
    "fft data/Resin/rec_2_lamps_top_off_fft.npy",                
    "fft data/Resin/square_1_lamp_left_on_fft.npy",
    "fft data/Resin/square_1_lamp_right_on_fft.npy",
    "fft data/Resin/square_1_lamp_top_on_fft.npy",                   
    "fft data/Resin/square_2_lamps_top_off_fft.npy",
    "fft data/Resin/square_3_lamps_fft.npy",
    "fft data/Resin/triangular_1_lamp_left_on_fft.npy",              
    "fft data/Resin/triangular_1_lamp_top_on_fft.npy",            
    "fft data/Resin/triangular_2_lamps_right_off_fft.npy",
    "fft data/Resin/triangular_2_lamps_top_off_fft.npy",
    "fft data/Resin/triangular_3_lamps_fft.npy"]

# define experimental test data set
TEST_DATA = [
    "fft data/Composite/new_1_lamp_left_on_fft.npy",
    "fft data/Composite/new_1_lamp_right_on_fft.npy",
    "fft data/Composite/new_1_lamp_top_on_fft.npy",
    "fft data/Resin/circular_2_lamps_right_off_fft.npy",
    "fft data/Resin/circular_2_lamps_top_off_fft.npy",
    "fft data/Resin/rec_1_lamp_top_on_fft.npy",
    "fft data/Resin/rec_3_lamps_fft.npy",
    "fft data/Resin/square_2_lamps_left_off_fft.npy",
    "fft data/Resin/square_2_lamps_right_off_fft.npy",
    "fft data/Resin/triangular_1_lamp_right_on_fft.npy",
    "fft data/Resin/triangular_2_lamps_left_off_fft.npy"]


# for loading pca data (comment out if loading pure fft data)
TRAIN_DATA = [f.replace('fft data', 'fftpca data') for f in TRAIN_DATA]
TEST_DATA = [f.replace('fft data', 'fftpca data') for f in TEST_DATA]
TRAIN_DATA = [f.replace('_fft', '_fftpca') for f in TRAIN_DATA]
TEST_DATA = [f.replace('_fft', '_fftpca') for f in TEST_DATA]

# define experimental labels
MASK_MAP = {
        "circular" : "Resin plates/circular.png",
        "rec" : "Resin plates/rec.png",
        "square" : "Resin plates/square.png",
        "tri" : "Resin plates/tri.png",
        "composite" : "Composite plate/composite plate mask.png"
} 

# define simulation training and test data
TRAIN_SIM_DATA = [
    "raw data/sim1_circular_heat5_5_1.npy",
    "raw data/sim1_circular_heat5_5_2.npy",
    "raw data/sim1_circular_heat5_5_1.npy",
    "raw data/sim1_circular_heat5_5_2.npy",
    "raw data/sim1_circular_heat5_5_3.npy",
    "raw data/sim1_circular_heat5_5_4.npy",
    "raw data/sim1_circular_heat5_5_5.npy",
    "raw data/sim1_circular_heat5_5_6.npy",
    "raw data/sim1_circular_heat5_5_7.npy",
    "raw data/sim1_circular_heat5_5_8.npy",
    "raw data/sim1_circular_heat5_5_9.npy",
    "raw data/sim1_circular_heat5_5_10.npy",
    "raw data/sim1_circular_heat5_5_11.npy",
    "raw data/sim1_circular_heat10_1.npy",
    "raw data/sim1_circular_heat10_2.npy",
    "raw data/sim1_circular_heat10_3.npy",
    "raw data/sim1_circular_heat10_4.npy",
    "raw data/sim1_circular_heat10_5.npy",
    "raw data/sim1_circular_heat10_6.npy",
    "raw data/sim1_circular_heat10_7.npy",
    "raw data/sim1_circular_heat10_8.npy",
    "raw data/sim1_circular_heat10_9.npy",
    "raw data/sim1_circular_heat10_10.npy",
]

TEST_SIM_DATA = [
    "raw data/sim1_circular_heat5_5_12.npy",
    "raw data/sim1_circular_heat5_5_13.npy",
    "raw data/sim1_circular_heat5_5_14.npy",
    "raw data/sim1_circular_heat5_5_15.npy",
    "raw data/sim1_circular_heat5_5_16.npy",
    "raw data/sim1_circular_heat5_5_17.npy",
    "raw data/sim1_circular_heat10_11.npy",
    "raw data/sim1_circular_heat10_12.npy",
    "raw data/sim1_circular_heat10_13.npy",
    "raw data/sim1_circular_heat10_14.npy",
    "raw data/sim1_circular_heat10_15.npy",
    "raw data/sim1_circular_heat10_16.npy",
    "raw data/sim1_circular_heat10_17.npy",
]

# define mask directory within Simulation Data directory
mask_dir = "masks/"

# evaluation function used across models
def evaluate(y_true, y_pred):
    # calculate evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

    return acc, precision, recall, f1, iou
