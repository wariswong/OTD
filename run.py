from app import create_app

app = create_app()

if __name__ == '__main__':
    # host='0.0.0.0' allows access from LAN
    # use_reloader=False: the auto-reloader watches the whole project tree,
    # including feature_shareload/FEASIBLE/ where shareload jobs write their
    # output — so it was restarting the whole app mid-job and killing every
    # shareload run within ~1s of starting. Restart the process manually
    # after editing code.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
