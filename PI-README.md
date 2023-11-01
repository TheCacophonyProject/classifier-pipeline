# PI classifier

This pacakge will install the cacophony pipeline tracking onto a RPI

You will also need to install these system packages in order to run the package
`sudo apt install python3-opencv libglib2.0-dev libgirepository1.0-dev libcairo2-dev libcairo2  python3-dbus build-essential libdbus-glib-1-dev libgirepository1.0-dev`

Once installed the tracking can be run using `pi_classifier`

The default config is installed with this package if you would like to override the config you can create a config file at
`/etc/cacophony/classifier.yaml`
