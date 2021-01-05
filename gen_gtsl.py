import gen_gts 
import gen_gtl 

if __name__ == "__main__":
    gt_file = 'training/gt_replaced_background.torch'
    savefolder_gts = 'training/gts/replaced_background/'
    savefolder_gtl = 'training/gtl/replaced_background/'

    dim1 = (360,360)
    dim2 = (45,45)
    sigma = 18
    gen_gts.generate_gts(gt_file, savefolder_gts, dim1, dim2, sigma)

    dim1 = (360,360)
    dim2 = (45,45)
    size = 10
    gen_gtl.gen_gtl_folder(gt_file, savefolder_gtl, dim1, dim2, size)