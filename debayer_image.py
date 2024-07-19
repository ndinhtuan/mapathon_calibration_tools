import cv2 
import numpy as np
import argparse

class ImageDebayer(object):

    def __init__(self) -> None:
        pass

    def debayer(self, img : np.ndarray) -> np.ndarray:
        
        return cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Argument for image debayer")

    parser.add_argument("--raw_image_path", type=str, help="Path to the raw image")
    parser.add_argument("--dst_image_path", type=str, help="Path to save the destination debayered image")
    parser.add_argument("--show_result", action="store_true", help="Option to show the debayered image")

    args = parser.parse_args()

    image_debayer = ImageDebayer()

    raw_img = cv2.imread(args.raw_image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    debayered_img = image_debayer.debayer(raw_img)

    if args.show_result:
        cv2.imshow("debayered image", debayered_img)
        cv2.waitKey(0)
    
    cv2.imwrite(args.dst_image_path, debayered_img)