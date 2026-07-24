@echo off
cd /d "d:\users\Lennart\code\mmv_h4tracks"
python "src\mmv_h4tracks\_tests\data\generate_numpy.py"
exit /b %ERRORLEVEL%
