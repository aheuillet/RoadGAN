from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.uix.modalview import ModalView
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.metrics import dp
from kivy.utils import get_color_from_hex

from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.list import MDList, OneLineAvatarIconListItem, ILeftBodyTouch, IRightBodyTouch
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.button import MDIconButton, MDRaisedButton
from kivymd.uix.dialog import BaseDialog
from kivymd.uix.spinner import MDSpinner
from kivymd.theming import ThemeManager
from kivymd.toast import toast
from kivymd import images_path
from kivymd.uix.bottomsheet import MDCustomBottomSheet

from few_shot_vid2vid.test import infer_images
from utils import recompose_video, decompose_video
import os
import sys
import shutil
from threading import Thread

KV='''

#:import utils kivy.utils

<SettingsScenario@BoxLayout>
    orientation: "vertical"
    size_hint_y: None
    height: "400dp"

    MDToolbar:
        title: 'Settings'
    
    ScrollView:

        MDList:
            OneLineAvatarIconListItem:
                text: "Weather conditions"
                ListDropDownLeft:
                    icon: "weather-cloudy"
                ListDropDownRight:
                    id: weather_cond
                    text: "clear"
                    on_release: app.weather_menu.open()
            
            OneLineAvatarIconListItem:
                text: "Urban style"
                ListDropDownLeft:
                    icon: "home-city"
                ListDropDownRight:
                    id: urban_style
                    text: "Berlin"
                    on_release: app.urban_style_menu.open()
            
            OneLineAvatarIconListItem:
                text: "Day time"
                ListDropDownLeft:
                    icon: "clock"
                ListDropDownRight:
                    id: day_time
                    text: "Day"
                    on_release: app.day_time_menu.open()
            
            OneLineIconListItem:
                text: "Exit"
                bg_color: utils.get_color_from_hex('#F44336')
                text_color: utils.get_color_from_hex('#FFFFFF')
                theme_text_color: 'Custom'
                on_release: app.close_settings()
                ListDropDownLeft:
                    text_color: utils.get_color_from_hex('#FFFFFF')
                    theme_text_color: 'Custom' 
                    icon: "exit-to-app"
        
        


BoxLayout:
    orientation: 'vertical'
    spacing: dp(5)

    MDToolbar:
        id: toolbar
        title: app.title
        left_action_items: [['menu', lambda x: None]]
        right_action_items: [['settings', lambda x: None], ['information', lambda x: None]]
        elevation: 10
        md_bg_color: app.theme_cls.primary_color


    FloatLayout:

        MDRoundFlatIconButton:
            text: "Choose input semantic video"
            icon: "folder"
            pos_hint: {'center_x': .5, 'center_y': .7}
            size_hint: dp(0.3), None
            on_release: app.file_manager_open()

        MDRoundFlatIconButton:
            text: "Choose output video save location"
            icon: "content-save"
            pos_hint: {'center_x': .5, 'center_y': .6}
            size_hint: dp(0.3), None
            on_release: app.file_manager_open(output=True)

        MDRoundFlatIconButton:
            text: "Change scenario"
            icon: "settings"
            pos_hint: {'center_x': .5, 'center_y': .5}
            size_hint: dp(0.3), None
            on_release: app.open_settings()

        MDFloatingActionButton:
            icon: "play"
            pos_hint: {'center_x': .5, 'center_y': .35}
            md_bg_color: app.theme_cls.primary_color
            elevation_normal: 11
            on_release: app.process_inference()
        
        MDSpinner:
            id: spinner
            size_hint: None, None
            size: dp(46), dp(46)
            pos_hint: {'center_x': .5, 'center_y': .2}
            active: False
                
        
'''


class ListDropDownLeft(ILeftBodyTouch, MDIconButton):
    pass

class ListDropDownRight(IRightBodyTouch, MDDropDownItem):
    pass
    

class RoadGANGUI(MDApp):
    title = "RoadGAN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.main_win = Builder.load_string(KV)
        self.manager_open = False
        self.manager = None
        self.settings_open = False
        self.input_path = None
        self.output_path = './'
        self.weather = "clear"
        self.day_time = "daylight"
        self.urban_style = "Stuttgart"
        self.weather_conditions = [{"icon": "weather-sunny", "text": "clear"}, {"icon": 'weather-fog', "text": "fog"}, {"icon": "weather-pouring", "text": 'rain'}, {"icon": "weather-snowy", "text": 'snow'}, {"icon": "weather-cloudy", "text": 'clouds'}]
        self.urban_styles = [{"icon": "home-city", "text": 'Stuttgart'}, {"icon": "home-city", "text": 'England'}, {"icon": "home-city", "text": 'France'}, {"icon": "home-city", "text": 'Boston'}, {"icon": "home-city", "text": 'China'}]
        self.day_times = [{"icon": "weather-sunset-up", "text":'dawn'}, {"icon": "weather-sunny", "text": 'daylight'}, {"icon": "weather-sunset", "text": 'dusk'}, {"icon": "weather-night", "text": 'night'}]
        self.inference_thread = Thread(target=self.launch_conversion)

    def build(self):
        self.theme_cls.primary_palette = "Amber"
        return self.main_win

    def open_settings(self):
        self.settings = MDCustomBottomSheet(screen=Factory.SettingsScenario()) 
        self.weather_menu = MDDropdownMenu(
            caller=self.settings.screen.ids.weather_cond,
            items=self.weather_conditions,
            position="auto",
            callback=self.update_weather_condition,
            width_mult=3,
        )
        self.urban_style_menu = MDDropdownMenu(
            caller=self.settings.screen.ids.urban_style,
            items=self.urban_styles,
            position="auto",
            callback=self.update_urban_style,
            width_mult=3,
        )
        self.day_time_menu = MDDropdownMenu(
            caller=self.settings.screen.ids.day_time,
            items=self.day_times,
            position="auto",
            callback=self.update_day_time,
            width_mult=3,
        )
        self.settings.screen.ids.weather_cond.set_item(self.weather)
        self.settings.screen.ids.day_time.set_item(self.day_time)
        self.settings.screen.ids.urban_style.set_item(self.urban_style) 
        self.settings.open()
    
    def close_settings(self):
        if self.settings:
            self.settings.dismiss()
    
    def update_weather_condition(self, instance):
        if instance.text != 'clear':
            self.weather = instance.text
        self.settings.screen.ids.weather_cond.set_item(instance.text)
    
    def update_day_time(self, instance):
        self.day_time = instance.text
        self.settings.screen.ids.day_time.set_item(self.day_time)
        
    def update_urban_style(self, instance):
        self.urban_style = instance.text
        self.settings.screen.ids.urban_style.set_item(self.urban_style) 

    def file_manager_open(self, output=False):
        if output:
            func = self.select_output_path
            path = self.output_path
        else:
            func = self.select_input_path
            path = self.input_path
        print(self.manager_open)
        if not self.manager_open:
            self.manager = MDFileManager(
                exit_manager=self.exit_manager, select_path=func)
            self.manager.ext = ['.mp4', '.avi']
            #self.manager.add_widget(self.file_manager)
            self.manager.show(os.path.dirname(path) if path else '/')  # output manager to the screen
            self.manager_open = True

    def select_input_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''

        self.exit_manager()
        self.input_path = path
        toast(path)
    
    def select_output_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''

        self.exit_manager()
        self.output_path = path
        toast(path)

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager.close()
        self.manager_open = False
    
    def process_inference(self):
        '''Called when the user clicks on the floating play button. 
        Launches the conversion using vid2vid and HAL.'''
        if self.inference_thread.is_alive():
            self.inference_thread.join()
        self.main_win.ids.spinner.active = True
        self.inference_thread.start()
    
    def launch_conversion(self):
        video_name = decompose_video(self.input_path)
        frame_dir_path = os.path.join(os.path.dirname(self.input_path), video_name)
        os.makedirs('./tmp')
        save_path = os.path.join('./tmp', video_name + "_converted")
        infer_images(frame_dir_path, os.path.abspath(self.select_style_img()), save_path)
        weather = "" if self.weather == "clear" else self.weather
        os.chdir('attribute_hallucination/')
        os.system("export MKL_SERVICE_FORCE_INTEL=1 && python generate_style.py --video_path " + os.path.join('..', save_path) + " --attributes " + self.day_time + " " + weather)
        os.system("export MKL_SERVICE_FORCE_INTEL=1 && python style_transfer.py --video_folder " + os.path.join('..', save_path))
        os.chdir('..')
        recompose_video('./tmp/' + video_name + "_converted_stylized/", os.path.join(self.output_path, video_name + "_converted.mp4"))
        shutil.rmtree('./tmp')
        self.main_win.ids.spinner.active = False
        toast('Inference finished!')
    
    def process_weather_conditions(self):
        return self.weather if self.weather != "" else "clear"
    
    def select_style_img(self):
        '''Return the path to the style image corresponding to the scenario chosen
        by the user.'''
        return os.path.join("inference/refs_img/images/", f"{self.urban_style.lower()}_{self.process_weather_conditions()}")

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device..'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.manager.back()
        return True

RoadGANGUI().run()
