import tkinter
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    min_width = 1050
    min_height = 500

    def __init__(self):
        super().__init__()

        self.title("Face Recognition")
        self.minsize(App.min_width, App.min_height)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # Call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (1 x 2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.left_frame = customtkinter.CTkFrame(master=self,
                                                 width=180)
        self.left_frame.grid(row=0, column=0, sticky="nswe", padx=(20, 0), pady=20)

        self.right_frame = customtkinter.CTkFrame(master=self)
        self.right_frame.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ left_frame ============

        # Configure grid layout (15 x 1)
        self.left_frame.grid_rowconfigure(0, minsize=10)  # empty row with minsize as spacing
        self.left_frame.grid_rowconfigure(6, weight=1)  # empty row as spacing
        self.left_frame.grid_rowconfigure(11, minsize=20)  # empty row as spacing
        self.left_frame.grid_rowconfigure(14, minsize=20)  # empty row with minsize as spacing

        # App title
        self.app_title = customtkinter.CTkLabel(master=self.left_frame,
                                                text="Face Recognition",
                                                text_font=("Open Sans Bold", -18))  # font name and size in px
        self.app_title.grid(row=1, column=0, pady=10, padx=10)

        # Import dataset
        self.dataset_label = customtkinter.CTkLabel(master=self.left_frame,
                                                    text="Import dataset",
                                                    text_font=("Open Sans", -14))  # font name and size in px
        self.dataset_label.grid(row=2, column=0, pady=0, padx=10)
        self.dataset_button = customtkinter.CTkButton(master=self.left_frame,
                                                      text="Open folder",
                                                      text_font=("Open Sans", -12),
                                                      command=button_event)
        self.dataset_button.grid(row=3, column=0, pady=5, padx=20)

        # Import test image
        self.test_label = customtkinter.CTkLabel(master=self.left_frame,
                                                 text="Import test image",
                                                 text_font=("Open Sans", -14))  # font name and size in px
        self.test_label.grid(row=4, column=0, pady=0, padx=10)
        self.test_button = customtkinter.CTkButton(master=self.left_frame,
                                                   text="Open image",
                                                   text_font=("Open Sans", -12),
                                                   command=self.get_test_image)
        self.test_button.grid(row=5, column=0, pady=5, padx=20)

        # Result
        self.result_label = customtkinter.CTkLabel(master=self.left_frame,
                                                   text="Result:",
                                                   text_font=("Open Sans Semibold", -14))  # font name and size in px
        self.result_label.grid(row=7, column=0, pady=0, padx=10)
        self.result_percentage_label = customtkinter.CTkLabel(master=self.left_frame,
                                                              text="",
                                                              text_font=(
                                                                  "Open Sans", -14))  # font name and size in px
        self.result_percentage_label.grid(row=8, column=0, pady=0, padx=10)

        # Execution time
        self.exec_label = customtkinter.CTkLabel(master=self.left_frame,
                                                 text="Execution time:",
                                                 text_font=("Open Sans Semibold", -14))  # font name and size in px
        self.exec_label.grid(row=9, column=0, pady=0, padx=10)
        self.exec_time_label = customtkinter.CTkLabel(master=self.left_frame,
                                                      text="",
                                                      text_font=(
                                                          "Open Sans", -14))  # font name and size in px
        self.exec_time_label.grid(row=10, column=0, pady=0, padx=10)

        # GUI theme
        self.theme_label = customtkinter.CTkLabel(master=self.left_frame, text="Select theme:", text_font=("Open Sans",
                                                                                                           -12))
        self.theme_label.grid(row=12, column=0, pady=0, padx=20, sticky="w")
        self.theme_options = customtkinter.CTkOptionMenu(master=self.left_frame,
                                                         values=["Dark", "Light", "System"],
                                                         text_font=("Open Sans", -12),
                                                         command=change_appearance_mode)
        self.theme_options.grid(row=13, column=0, pady=5, padx=20, sticky="w")

        # ============ right_frame ============

        # Configure grid layout (5x2)
        self.right_frame.rowconfigure((0, 1, 2, 3), weight=1)
        self.right_frame.rowconfigure(0, weight=0)
        self.right_frame.columnconfigure((0, 1), weight=1)

        # Test image
        self.test_image_label = customtkinter.CTkLabel(master=self.right_frame,
                                                       text="Test image",
                                                       text_font=(
                                                           "Open Sans", -16))  # font name and size in px
        self.test_image_label.grid(row=1, column=0, pady=(20, 0), padx=10)

        self.test_image_frame = customtkinter.CTkFrame(master=self.right_frame)
        self.test_image_frame.grid(row=2, column=0, pady=0, padx=(20, 0), sticky="n")
        self.test_image_frame.rowconfigure((0, 2), minsize=20)
        self.test_image_frame.columnconfigure((0, 2), minsize=20)

        self.test_image_button = customtkinter.CTkButton(master=self.test_image_frame,
                                                         text="",
                                                         width=323,
                                                         height=323,
                                                         fg_color=("#C0C2C5", "#343638"),
                                                         borderwidth=0,
                                                         hover=False)
        self.test_image_button.grid(row=1, column=1, pady=0, padx=0, sticky="nswe")

        # Closest result image
        self.closest_image_label = customtkinter.CTkLabel(master=self.right_frame,
                                                          text="Closest result",
                                                          text_font=(
                                                              "Open Sans", -16))  # font name and size in px
        self.closest_image_label.grid(row=1, column=1, pady=(20, 0), padx=10)

        self.closest_image_frame = customtkinter.CTkFrame(master=self.right_frame)
        self.closest_image_frame.grid(row=2, column=1, pady=0, padx=20, sticky="n")
        self.closest_image_frame.rowconfigure((0, 2), minsize=20)
        self.closest_image_frame.columnconfigure((0, 2), minsize=20)

        # closest_image = Image.open("tobi.jpg")
        # closest_image = closest_image.resize((400, 400))
        # closest_image = ImageTk.PhotoImage(closest_image)
        self.closest_image_button = customtkinter.CTkButton(master=self.closest_image_frame,
                                                            text="",
                                                            width=323,
                                                            height=323,
                                                            fg_color=("#C0C2C5", "#343638"),
                                                            borderwidth=0,
                                                            hover=False)
        self.closest_image_button.grid(row=1, column=1, pady=0, padx=0, sticky="nswe")

        # Set default values
        self.theme_options.set("Dark")

    def on_closing(self):
        self.destroy()

    def get_test_image(self):
        file_type = [("JPG File", "*.jpg")]
        test_image_file_name = tkinter.filedialog.askopenfilename(title='Open image', filetypes=file_type)

        test_image = Image.open(test_image_file_name)
        test_image = test_image.resize((400, 400))
        test_image = ImageTk.PhotoImage(test_image)
        self.test_image_button.configure(image=test_image)


def button_event():
    print("Button pressed")


def change_appearance_mode(new_appearance_mode):
    customtkinter.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = App()
    app.mainloop()
