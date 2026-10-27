# mac/

Put the KoboldCPP program for macOS (Apple Silicon) here: `koboldcpp-mac-arm64`.

Download it from <https://github.com/LostRuins/koboldcpp/releases> (it is not stored in this repository). Then run `start_model.command`. See the main README, sections 4 and 5, for the full steps and for troubleshooting.

*Troubleshooting:*
- **macOS:** open Terminal and run `chmod +x mac/koboldcpp-mac-arm64 mac/start_model.command` once, then double-click `start_model.command`. MacOS blocks koboldcpp-mac-arm64, open *System Settings → Privacy & Security* and allow it, or right-click the file and choose *Open*. Then rerun the command to run the localhost. 
