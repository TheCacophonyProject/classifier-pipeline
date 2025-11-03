def pi_classify():
    from piclassifier.piclassify import main

    main()


def serve_model():
    from piclassifier.servemodel import main

    main()


def postprocess_watcher():
    from piclassifier.postprocess import main

    main()


def dbus_listener():
    from piclassifier.dbuslistener import main

    main()
