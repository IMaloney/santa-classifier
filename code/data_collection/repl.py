import cmd2
from data_collection.data_collection import collect_data
from data_collection.raw_processing_utils import find_and_remove_similar_images, rename_images
from data_collection.face_recognition import find_images_with_more_than_one_face, find_images_with_no_faces
from data_collection.data_collection_utils import prompt_delete, shuffle_images
from shared.utils import random_string
from shared.constants import THRESHOLD, IMAGE_SIZE, NUM_IMAGES

class DataCollectionCmd(cmd2.Cmd):
    prompt = "> "

    def __init__(self):
        super().__init__()

    remove_similar_images_parser = cmd2.Cmd2ArgumentParser()
    remove_similar_images_parser.add_argument("-f", "--folder_path", nargs="?", help="Folder path containing images", required=True)
    remove_similar_images_parser.add_argument("-t", "--threshold", nargs="?", default=THRESHOLD, type=int, help=f"Threshold value for similarity (default is {THRESHOLD})", required=False)
    @cmd2.with_argparser(remove_similar_images_parser)
    def do_remove_dups(self, args):
        """
        Remove visually similar images from the specified folder with the specified threshold.
        Usage: remove_similar_images -f [folder_path] -t [threshold]
        """
        find_and_remove_similar_images(args.folder_path, args.threshold, True)

    shuffle_images_parser = cmd2.Cmd2ArgumentParser()
    shuffle_images_parser.add_argument("-f", "--folder_path", nargs="?", help="Folder path containing images", required=True)
    @cmd2.with_argparser(shuffle_images_parser)
    def do_shuffle_images(self, args):
        """
        Shuffles images in directory
        Usage: shuffle_images -f [folder_path]
        """
        shuffle_images(args.folder_path)
        
    rename_images_parser = cmd2.Cmd2ArgumentParser()
    rename_images_parser.add_argument("-f", "--folder_path", nargs="?", help="Folder path containing images", required=True)
    rename_images_parser.add_argument("-n", "--new_name", nargs="?", help="New name for the images", required=True)
    @cmd2.with_argparser(rename_images_parser)
    def do_rename_images(self, args):
        """
        Rename all images in the specified folder with the new name, appending the file count to the end.
        Usage: rename_images -f [folder_path] -n [new_name]
        """
        random_name = random_string(16)
        temp_name = f"temp_name_{random_name}"
        rename_images(args.folder_path, temp_name)
        rename_images(args.folder_path, args.new_name, True)
        
    data_collection_parser = cmd2.Cmd2ArgumentParser()
    data_collection_parser.add_argument("-q", "--query", nargs="*", help="Search Query", required=True)
    data_collection_parser.add_argument("-s", "--size", nargs="?", default=IMAGE_SIZE, type=int, help="target size of the downloaded image")
    data_collection_parser.add_argument("-f", "--folder", nargs="?", required=True, help="Folder path to save images")
    data_collection_parser.add_argument("-i", "--images", nargs="?", type=int, default=NUM_IMAGES, help="Number of images to download (will cap itself at 200)")
    @cmd2.with_argparser(data_collection_parser)
    def do_collect_data(self, args):
        """
        Queries the google search api for images, crops them to a specific size, then saves them in the folder. 
        Usage: collect_data -q [query] -s [size] -f [folder_path] -i [num images]
        """
        collect_data(args.query, args.size, args.folder, args.images)
        
    face_detection_parser = cmd2.Cmd2ArgumentParser()
    face_detection_parser.add_argument("-f", "--folder_path", nargs="?", required=True, help="folder path")
    @cmd2.with_argparser(face_detection_parser)
    def do_face_detect(self, args):
        """
        Finds the all images with more than one face present
        Usage: face_detect -f [folder_path]
        """
        images = find_images_with_more_than_one_face(args.folder_path)
        prompt_delete(images)
        
    zero_face_detection_parser = cmd2.Cmd2ArgumentParser()
    zero_face_detection_parser.add_argument("-f", "--folder_path", nargs="?",required=True, help="folder path")
    @cmd2.with_argparser(zero_face_detection_parser)
    def do_no_face_detect(self, args):
        """
        Finds the all images zero faces present
        Usage: no_face_detect -f [folder_path]
        """
        images = find_images_with_no_faces(args.folder_path)
        prompt_delete(images)


def start_repl():
    print("The Data Collection repl:")
    print("\tEasily query images to build up a dataset")
    print("\tAlso allows for duplicate removal, cropping, and folder maintanence")
    app = DataCollectionCmd()
    app.cmdloop()