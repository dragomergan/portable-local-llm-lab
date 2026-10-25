@echo off
setlocal EnableDelayedExpansion
rem Start KoboldCPP (Windows) with a model that you choose.
rem
rem Put your .gguf model files in the "models" folder, next to the folder that holds
rem this script. The script lists the chat models it finds and asks which one to run.
rem A file whose name contains "embed" is used as the embedding model (needed for RAG).
rem Files whose name contains "mmproj" (image add-ons) are ignored.
rem
rem To skip the menu:  start_model.bat <model-file-name.gguf>

rem Settings you may want to change
set "PORT=5001"
set "THREADS=8"
set "CONTEXT=8192"

rem Paths are relative to this script.
set "HERE=%~dp0"
set "MODELS=%HERE%..\models"
set "KOBOLD=%HERE%koboldcpp.exe"

if not exist "%KOBOLD%" (
  echo KoboldCPP program not found: %KOBOLD%
  echo Download koboldcpp.exe from https://github.com/LostRuins/koboldcpp/releases and put it in this folder.
  goto :end
)

rem Find the models. Embedding models and "mmproj" files are not offered as chat models.
set COUNT=0
set "EMBED="
for /f "delims=" %%F in ('dir /b /a-d "%MODELS%\*.gguf" 2^>nul') do (
  set "NAME=%%F"
  set "ISEMBED="
  set "ISMMPROJ="
  echo !NAME! | findstr /i /c:"embed" >nul && set "ISEMBED=1"
  echo !NAME! | findstr /i /c:"mmproj" >nul && set "ISMMPROJ=1"
  if defined ISEMBED (
    if not defined EMBED set "EMBED=%%F"
  ) else if not defined ISMMPROJ (
    set /a COUNT+=1
    set "MODEL_!COUNT!=%%F"
  )
)

set "SELECTED="
if not "%~1"=="" (
  if exist "%MODELS%\%~1" (
    set "SELECTED=%~1"
  ) else (
    echo Model not found: %MODELS%\%~1
    goto :end
  )
  goto :launch
)

if %COUNT%==0 (
  echo No chat model ^(.gguf^) found in %MODELS%
  echo Download one ^(see the README^), put it in that folder, and run this script again.
  goto :end
)
if %COUNT%==1 (
  set "SELECTED=!MODEL_1!"
  echo Only one model found: !SELECTED!
  goto :launch
)

echo.
echo Available models:
for /l %%I in (1,1,%COUNT%) do echo   %%I^) !MODEL_%%I!
echo.
:ask
set "CHOICE="
set /p "CHOICE=Type the number of the model and press Enter: "
for /l %%I in (1,1,%COUNT%) do (
  if "!CHOICE!"=="%%I" set "SELECTED=!MODEL_%%I!"
)
if not defined SELECTED (
  echo Please type a number between 1 and %COUNT%.
  goto :ask
)

:launch
echo.
echo Starting with model: %SELECTED%
if defined EMBED (
  echo Embedding model:     %EMBED%
  echo.
  "%KOBOLD%" --model "%MODELS%\%SELECTED%" --embeddingsmodel "%MODELS%\%EMBED%" --port %PORT% --threads %THREADS% --contextsize %CONTEXT%
) else (
  echo No embedding model found, so RAG features will not work ^(put a file with "embed" in its name into the models folder^).
  echo.
  "%KOBOLD%" --model "%MODELS%\%SELECTED%" --port %PORT% --threads %THREADS% --contextsize %CONTEXT%
)

:end
echo.
pause
endlocal
