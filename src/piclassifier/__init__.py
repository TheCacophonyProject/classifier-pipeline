def pi_classify():
    from config.thermalconfig import ThermalConfig

    parse_file = None
    thermal_config = ThermalConfig.load_from_file()
    parse_file = None
    if (
        thermal_config.recorder.instant_classify
        and thermal_config.recorder.rec_window.inside_window()
    ):
        from mediumpower.mediumpower import main

        parse_file = main(thermal_config)
    from piclassifier.piclassify import main

    main(thermal_config, parse_file=parse_file)


def serve_model():
    from piclassifier.servemodel import main

    main()


def postprocess_watcher():
    from piclassifier.postprocess import main

    main()


def dbus_listener():
    from piclassifier.dbuslistener import main

    main()
