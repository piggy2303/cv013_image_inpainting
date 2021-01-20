from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(
    file_id='1LVyaCQS6xhyIwqx-PeUs6IGV0-pXJrin',
    dest_path='pretrained_model/deepfillv2_WGAN_G_epoch40_batchsize4.pth'
)
print("done")
