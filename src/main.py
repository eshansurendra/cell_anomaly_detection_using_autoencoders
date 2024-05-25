from data import load_data
from train import build_and_train_model
from evaluate import calculate_density_and_recon_error, check_anomaly
from visualize import plot_loss

def main():
    train_generator, validation_generator, anomaly_generator = load_data()

    model, encoder_model, history = build_and_train_model(train_generator, validation_generator)
    
    plot_loss(history)
    
    train_batch = train_generator.next()[0]
    anomaly_batch = anomaly_generator.next()[0]

    uninfected_values = calculate_density_and_recon_error(train_batch, model, encoder_model)
    anomaly_values = calculate_density_and_recon_error(anomaly_batch, model, encoder_model)

    import glob
    para_file_paths = glob.glob('cell_images2/parasitized/images/*')
    uninfected_file_paths = glob.glob('cell_images2/uninfected_train/images/*')

    # Anomaly image verification
    num = random.randint(0, len(para_file_paths) - 1)
    check_anomaly(para_file_paths[num], model, encoder_model, uninfected_values, anomaly_values)

    # Good/normal image verification
    num = random.randint(0, len(para_file_paths) - 1)
    check_anomaly(uninfected_file_paths[num], model, encoder_model, uninfected_values, anomaly_values)

if __name__ == "__main__":
    main()
