# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# import owl_vit_minimal_example      # 22222

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",      # 00000
        default="Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",           # 22222
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        # default=['/home/yuan/Mani-GPT/camera_capture/1.png'],                                 # 11111
        default=['/home/yuan/Mani-GPT/camera_capture/6.jpeg'],                                  # kitchen
        # default=owl_vit_minimal_example.filename_input,                                       # 22222
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        # default="/home/yuan/Mani-GPT/camera_capture/1_object_detection_lvis.jpeg",            # lvis
        default="/home/yuan/Mani-GPT/camera_capture/6_object_detection_custom.jpeg",            # custom
        # default=owl_vit_minimal_example.filename_output,                                      # 11111
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        # default="lvis",
        # default="openimages",                                                                 # 11111
        default="custom",                                                                       # 11111
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],  
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        # default="",
        default = 
            "Knife, Cutting board, Spoon, Fork, Plate, Bowl, Cup, Mug, Glass, Pan, Pot, Lid, Spatula, Whisk, Tongs, Grater, Peeler, Strainer, Rolling pin, "
            "Baking sheet, Oven mitt, Measuring spoon, Blender, Food processor, Mixer, Toaster, Microwave, Stove, Oven, Refrigerator, Freezer, Can opener, "
            "Bottle opener, Corkscrew, Timer, Thermometer, Scale, Ladle, Skillet, Saucepan, Pressure cooker, Slow cooker, Rice cooker, Steamer, Wok, Grill, Sandwich maker, "
            "Coffee maker, Tea kettle, Dish rack, Dish towel, Dish soap, Sponge, Scrubber, Garbage disposal, Trash can, Recycling bin, Apron, Trivet, Potholder, Salt shaker, "
            "Pepper grinder, Spice rack, Storage container, Plastic wrap, Aluminum foil, Parchment paper, Wax paper, Basting brush, Pastry brush, Pastry bag, Cookie cutter, "
            "Ice tray, Juicer, Zester, Mortar, Pestle, Mandoline, Salad spinner, Egg slicer, Garlic press, Meat tenderizer, Meat thermometer, Baster, Skimmer, Slotted spoon, "
            "Soup ladle, Pizza cutter, Pizza stone, Cheese grater, Cheese slicer, Funnel, Melon baller, Nutcracker, Rolling pin, Sifter, Stand mixer, Waffle maker, Whisk attachment, "
            "Butter dish, Casserole dish, Carving knife, Cheese board, Cleaver, Coffee grinder, Cookie sheet, Cooling rack, Corn stripper, Cutting mat, Dough scraper, Egg beater, "
            "Egg, egg cup, Egg timer, Fish scaler, Flour sifter, Food mill, Frying pan, Garlic keeper, Garlic roaster, Gravy boat, Herb chopper, Ice bucket, Ice cream scoop, "
            "Ice pick, Ice tongs, Julienne peeler, Kettle, Kitchen scale, Ladle holder, Lemon squeezer, Mandoline slicer, Meat grinder, Meat slicer, Muffin tin, "
            "Oil dispenser, Pastry blender, Pastry wheel, Pie plate, Pizza pan, Potato masher, Ravioli cutter, Roasting pan, Salad bowl, Salad fork, Salad server, Salad tongs, "
            "Sauce boat, Sauce ladle, Serrated knife, Serving fork, Serving spoon, Slicer, Slotted spatula, Soup bowl, Soup spoon, Sugar bowl, Tea strainer, Thermos, "
            "Tomato slicer, Trivet mat, Turkey baster, Vegetable peeler, Water pitcher, Whipping siphon, Wooden spoon, Zester grater, Apple corer, Baking mold, Basting syringe, "
            "Blow torch, Bread basket, Bread knife, Butter knife, Cake stand, Cake tester, Cheese plane, Chopsticks, Citrus press, Cleaver knife, Cookie press, Crepe pan, "
            "Crumb tray, Deep fryer, Deviled egg plate, Dough whisk, Electric knife, Espresso machine, Fish spatula, Flour scoop, Food dehydrator, French press, Fry basket, "
            "Garlic mincer, Ginger grater, Griddle, Hamburger press, Honey dipper, Ice cream maker, Ice cube tray, Jam funnel, Jelly mold, Juicer machine, Kabob skewer, "
            "Kitchen timer, Knife block, Knife sharpener, Lobster cracker, Mandoline cutter, Measuring jug, Melon slicer, Mesh strainer, Muffin pan, Nutmeg grater, Onion chopper, "
            "Oven thermometer, Pancake turner, Pastry bag tips, Pepper mill, Pie server, Pineapple corer, Popcorn maker, Popsicle mold, Poultry shears, Ravioli stamp, Ricer, "
            "Roasting rack, Seafood cracker, Serrated bread knife, Silicone baking mat, Slicing knife, Spice jar, Spiralizer, Steamer basket, "
            "Sushi mat, Tea infuser, Tea pot, Thermometer probe, Truffle shaver, Vegetable chopper, Waffle iron, Wheat grinder, Whipped cream dispenser, Wine aerator, "
            "Wine decanter, Wine opener, Wooden salad bowl, Zester tool, Artichoke, Arugula, Asparagus, Eggplant, Beet, Bell pepper, Bok choy, Broccoli, Broccoli rabe, "
            "Brussels sprouts, Cabbage, Carrot, Cauliflower, Celery, Chard, Chayote, Chicory, Collard greens, Corn, Cucumber, Daikon, Dandelion greens, Endive, Fennel, "
            "Garlic, Ginger, Green bean, Green onion, Jicama, Kale, Kohlrabi, Leek, Lettuce, Mushroom, Mustard greens, Nappa cabbage, Okra, Onion, Parsnip, Pea, Potato, "
            "Pumpkin, Radicchio, Radish, Rhubarb, Rutabaga, Shallot, Spinach, Squash, Sweet potato, Tomato, Turnip, Watercress, Zucchini, Alfalfa sprouts, Bamboo shoots, "
            "Bean sprouts, Bitter melon, Butternut squash, Celeriac, Chili pepper, Courgette, Edamame, Fiddlehead fern, Horseradish, Jerusalem artichoke, Kohlrabi greens, "
            "Lotus root, Malabar spinach, Mizuna, Mung bean sprouts, Napa cabbage, Nopales, Parsley root, Poblano pepper, Purslane, Radish sprouts, Red cabbage, Scallion, "
            "Snow pea, Sorrel, Soybean sprouts, Sugar snap pea, Taro, Tomatillo, Turmeric, Wax bean, White asparagus, Yellow squash, Acorn squash, "
            "Amaranth, Anise, Arrowroot, Arugula, Baby corn, Black-eyed pea, Broccoflower, Broccolini, Butterhead lettuce, Cactus leaf, Cardoon, Celtuce, "
            "Chard stems, Chaya, Chervil, Chinese broccoli, Chinese chives, Chinese eggplant, Chinese mustard, Chinese spinach, Choy sum, Cilantro, Coral lettuce, Corn salad, "
            "Courgette flower, Dulse, Epazote, Fava bean, French bean, Frisée, Gai lan, Galangal, Garden cress, Gem squash, Golden beet, Haricot bean, "
            "Hijiki, Hubbard squash, Hyacinth bean, Iceberg lettuce, Kabocha squash, Kidney bean, Lacinato kale, Lamb's lettuce, Land cress, Lima bean, Mallow, Manoa lettuce, "
            "Molokhia, New Zealand spinach, Orach, Oyster plant, Pak choy, Perilla, Pinto bean, Poblano, Pumpkin flower, Purple cabbage, Red leaf lettuce, Romanesco, Roselle, "
            "Runner bean, Samphire, Scorzonera, Shiso, Snap pea, Snow pea shoot, Sorrel, Speckled lettuce, Spelt, Spring onion, Sugar beet, Summer squash, Tatsoi, Tepary bean, "
            "Tigernut, Topinambur, Trombocino squash, Tumbleweed, Water chestnut, Water spinach, White radish, Winged bean, Winter melon, Winter squash, Yardlong bean, "
            "Yellow bean, Yellow pea, Apple, Apricot, Avocado, Banana, Blackberry, Blackcurrant, Blueberry, Boysenberry, Cantaloupe, Cherry, Clementine, Coconut, Cranberry, "
            "Currant, Date, Dragonfruit, Durian, Elderberry, Feijoa, Fig, Goji berry, Gooseberry, Grape, Grapefruit, Guava, Honeydew, Huckleberry, Jabuticaba, Jackfruit, "
            "Jujube, Kiwifruit, Kumquat, Lemon, Lime, Longan, Loquat, Lychee, Mandarin, Mango, Mangosteen, Melon, Mulberry, Nectarine, Orange, Papaya, Passionfruit, Peach, "
            "Pear, Persimmon, Pineapple, Plum, Pomegranate, Pomelo, Quince, Raspberry, Redcurrant, Salak, Satsuma, Starfruit, Strawberry, Tamarillo, Tangerine, "
            "Watermelon, White currant, Yellow watermelon, Açaí, Acerola, Ambarella, Babaco, Bael, Banana passionfruit, Barberry, Bayberry, "
            "Bilberry, Biriba, Black sapote, Blood orange, Breadfruit, Buddha's hand, Cacao, Canistel, Carambola, Cempedak, Chayote, Chokeberry, Cloudberry, "
            "Cucumber, Cupuacu, Damson, Desert lime, Dodder laurel, Emblic, Finger lime, Gac, Galia melon, Genip, Giant granadilla, Granadilla, Greengage, "
            "Grosella, Hackberry, Hala fruit, Horned melon, Illawarra plum, Imbe, Boxed milk",
        # default="handle",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # default=[],                                                                                                     # 00000
        default=["MODEL.WEIGHTS","/home/yuan/Mani-GPT/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],    # 11111
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False



def detect_exist_objects():                                                              # 11111
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)

    if args.input:
        if len(args.input) == 1:    
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            # print("After processing:", args.input)
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output, labels_onlyclass = demo.run_on_image(img)    # 11111
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        if args.webcam == "screen":
            cam = ScreenGrab()
        else:
            cam = cv2.VideoCapture(int(args.webcam))
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()




    # print(labels_onlyclass)

    # labels_onlyclass = Detic.demo.get_labels_onlyclass()
    n = len(labels_onlyclass)
    exist_object_prompt = "You can see"
    for i, exist_object in enumerate(labels_onlyclass):
        if i == n-2:
            exist_object_prompt += exist_object + " and"
        elif i == n-1:
            exist_object_prompt += exist_object
        else:
            exist_object_prompt += exist_object + ","
    exist_object_prompt += " on the table."
    print(exist_object_prompt)

    # return labels_onlyclass
    return exist_object_prompt                                                            # 11111



detect_exist_objects()




# filename_input = os.path.join('/home/yuan/Mani-GPT/camera_capture/1.png')
# path, filename = os.path.split(filename_input)
# new_filename = filename.replace('.png', '_openimages.png')
# filename_output = os.path.join(path, new_filename)