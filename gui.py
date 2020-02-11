from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.uix.modalview import ModalView

from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.list import MDList, OneLineIconListItem
from kivymd.uix.dropdownitem import MDDropdownItem
from kivymd.theming import ThemeManager
from kivymd.toast import toast


Builder.load_string('''

<ExampleFileManager@BoxLayout>
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
            pos_hint: {'center_x': .5, 'center_y': .6}
            size_hint: dp(0.3), None
            on_release: app.file_manager_open()
        
        MDRoundFlatIconButton:
            text: "Choose output video save location"
            icon: "content-save"
            pos_hint: {'center_x': .5, 'center_y': .5}
            size_hint: dp(0.3), None
            on_release: app.file_manager_open(output=True)
        
        MDRoundFlatIconButton:
            text: "Change scenario"
            icon: "settings"
            pos_hint: {'center_x': .5, 'center_y': .4}
            size_hint: dp(0.3), None
            on_release: app.file_manager_open(output=True)
        
        MDFloatingActionButton:
            icon: "play"
            pos_hint: {'center_x': .5, 'center_y': .3}
            md_bg_color: app.theme_cls.primary_color
            elevation_normal: 11

    
''')


class RoadGANGUI(MDApp):
    title = "RoadGAN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.manager = None
        self.settings = None
        self.settings_open = False
        self.input_path = None
        self.output_path = None

    def build(self):
        self.theme_cls.primary_palette = "Amber"
        return Factory.ExampleFileManager()

    def open_settings(self):
         if not self.settings:
            self.settings = ModalView()
            self.scenario_selector = MDList()
            self.settings.add_widget(self.scenario_selector)
            weather = OneLineIconListItem(text="Weather conditions")
            weather.add_widget()
            self.scenario_selector.add_widget(MDDropdownItem())
            self.manager.add_widget(self.file_manager)
            self.file_manager.show('/')  # output manager to the screen
        self.manager_open = True
        self.manager.open()

    def file_manager_open(self, output=False):
        if output:
            func = self.select_output_path
        else:
            func = self.select_input_path
        if not self.manager:
            self.manager = ModalView(size_hint=(1, 1), auto_dismiss=False)
            self.file_manager = MDFileManager(
                exit_manager=self.exit_manager, select_path=func)
            self.manager.add_widget(self.file_manager)
            self.file_manager.show('/')  # output manager to the screen
        self.manager_open = True
        self.manager.open()

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
        self.input_path = path
        toast(path)

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager.dismiss()
        self.manager_open = False

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device..'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True


RoadGANGUI().run()