
def paint_arr_to_jpg(arr, filename='test.jpg', do_norm=True):
    norm = mcolors.Normalize(vmin=arr.min(), vmax=arr.max())
    cmap = plt.get_cmap('CMRmap') 
    img = cmap(norm(arr.T))
    imgsave = (img * 255).astype(np.uint8)[:,:,0:3]
    im = Image.fromarray(imgsave)
    im.save(filename)