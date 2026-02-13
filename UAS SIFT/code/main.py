import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

"""
    - function untuk load gambar dan mengubah jadi grayscale
"""
def load_images(path1, path2):
    print(f"load images path1: {path1} dan path2: {path2}")

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None:
        print(f"gambar 1 error ketika di load {path1}")
        sys.exit()

    if img2 is None:
        print(f"gambar 2 error ketika di load {path2}")
        sys.exit()

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print("berhasil konversi gambar")

    return img1, img1_gray, img2, img2_gray

"""
    - mencari keypoints dan descriptor SIFT
"""
def sift_keypoint(img_gray, img_color, output_filename):
    print(f"sift_keypoint {output_filename}")

    sift = cv2.SIFT_create()

    #deteksi keypoint dan hitung descriptor
    keypoint, descriptors = sift.detectAndCompute(img_gray, None)
    print(f"jumlah keypoint: {len(keypoint)}")

    #draw keypoiny
    img_sift = cv2.drawKeypoints(
        img_color,
        keypoint,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS #ntuk menampilkan lingkaran size dan garis 
    )

    output_path = f"./results/{output_filename}"
    cv2.imwrite(output_path, img_sift)
    print(f"sift keypoint berhasil {output_path}")

    return keypoint, descriptors

"""
    - DIffirent of Gaussion (DoG)
"""
def dog_visualize(img_gray, output_filename="dog.jpg"):
    print(f"DoG: {output_filename}")

    sigma1 = 1.6
    k = 1.6
    sigma2 = sigma1 * k

    blur1 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma1)
    blur2 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma2)

    dog = cv2.subtract(blur1, blur2)

    plt.figure(figsize=(15, 5))

    # blur sigma 1
    plt.subplot(1, 3, 1)
    plt.imshow(blur1, cmap='gray')
    plt.title(f'Gaussion Blur (sigma={sigma1})')
    plt.axis('off')

    # blur sigma 2
    plt.subplot(1, 3, 2)
    plt.imshow(blur2, cmap='gray')
    plt.title(f'Gaussion Blur (sigma={sigma2})')
    plt.axis('off')

    # Dog
    plt.subplot(1, 3, 3)
    plt.imshow(dog, cmap='gray')
    plt.title(f'Different of Gaussian')
    plt.axis('off')

    output_path = f"./results/{output_filename}"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"DoG berhasil disimpan => {output_path}")
    
"""
    - mencocokan fitur kedua gambar
"""
def match_feature(img1, kp1, des1, img2, kp2, des2, output_filename="matches.jpg"):
    print(f"mulai match_Feature {output_filename}")

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    good_matches_list_for_draw = []

    ration_tresh = 0.75

    # m = match terbaik pertama, n = match terbaik kedua
    for m, n in matches:
        if  m.distance < n.distance * ration_tresh:
            good_matches.append(m)
            good_matches_list_for_draw.append([m])

    print(f"mathces sebelumnya {len(matches)}, good matches {len(good_matches)}")

    # mathcing visualize
    img_matches = cv2.drawMatchesKnn(
        img1, kp1,
        img2, kp2,
        good_matches_list_for_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    output_path = f"./results/{output_filename}"
    cv2.imwrite(output_path, img_matches)
    print(f"match_feature disipan => {output_path}")

    return good_matches


"""
    - function untuk membuat panoramic
"""
# def create_panoramic(kp1, kp2, good_matches, img1, img2, output_filename="panoramic.jpg"):
#     print(f"mulai create_panoramic {output_filename}")

#     if (len(good_matches) < 4):
#         print(f"Error: kurang dari 4")
#         return
    
#     #ekstrak koordinat dari good_matches
#     src_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)

#     # hitung matriks
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     if M is None: 
#         print(f"Error: gagal menemukan homography")
#         return
    
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     canvas_width = w1 * w2
#     canvas_height = max(h1, h2)

#     print(f"canvas barus canvas_width {canvas_width} canvas_height {canvas_height}")

#     #wrapping / melengkung
#     panorama = cv2.warpPerspective(img2, M, (canvas_width, canvas_height))

#     #stich / menjahit
#     panorama[0:h1, 0:w1] = img1

#     output_path = f"./results/{output_filename}"
#     cv2.imwrite(output_path, panorama)
#     print(f"Panorama berhasil disimpan => {output_path}")
def create_panoramic(kp1, kp2, good_matches, img1, img2, output_filename="panorama.jpg"):
    print(f"Mulai membuat Panorama Robust {output_filename}...")

    if len(good_matches) < 4:
        print("Error: Kurang dari 4 titik matching.")
        return

    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("Gagal menghitung Homography.")
        return

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    pts_img2_transformed = cv2.perspectiveTransform(pts_img2, M)

    pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    all_pts = np.concatenate((pts_img1, pts_img2_transformed), axis=0)

    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    translation_dist = [-xmin, -ymin]
    
    H_translation = np.array([[1, 0, translation_dist[0]], 
                              [0, 1, translation_dist[1]], 
                              [0, 0, 1]])

    final_M = H_translation.dot(M)

    output_w = xmax - xmin
    output_h = ymax - ymin
    
    print(f"  -> Ukuran kanvas final: {output_w}x{output_h} (Wajar)")
    
    panorama = cv2.warpPerspective(img2, final_M, (output_w, output_h))

    t_x = translation_dist[0]
    t_y = translation_dist[1]
    
    panorama[t_y:h1+t_y, t_x:w1+t_x] = img1

    output_path = f"./results/{output_filename}"
    cv2.imwrite(output_path, panorama)
    print(f"Panorama berhasil disimpan => {output_path}")

if __name__ == "__main__":
    file_gambar_1 = "./images/1.jpeg"
    file_gambar_2 = "./images/2.jpeg"

    try:
        #LOAD IMAGES
        img1_c, img1_g, img2_c, img2_g = load_images(file_gambar_1, file_gambar_2)

        #SIFT KEYPOINT
        kp1, des1 = sift_keypoint(img1_g, img1_c, "img1_keypoints.jpg")
        kp2, des2 = sift_keypoint(img2_g, img2_c, "img2_keypoints.jpg")

        #DOG
        dog_visualize(img1_g, "dog_visualization.jpg")

        #FEATURE MATCHING
        good_matches = match_feature(
            img1_c, kp1, des1, 
            img2_c, kp2, des2,
            "matches.jpg"
        )

        #PANORAMIC
        if len(good_matches) > 4:
            create_panoramic(
                kp1, kp2, good_matches, 
                img1_c, img2_c, 
                "panorama_result_2.jpg"
            )
        else:
            print("WARNING: Matches terlalu sedikit harus lebih dari 4.")

        print("DONE")

    except Exception as e:
        print(f"Error : {e}")
        import traceback
        traceback.print_exc()