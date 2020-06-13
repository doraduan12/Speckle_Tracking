pyinstaller -D -i ../img/icon.ico -w ../app.py
copy ..\setting.json .\dist\app\
copy ..\setting_default.json .\dist\app\