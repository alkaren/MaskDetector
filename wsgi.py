from app import create_app
from flask_toastr import Toastr

app = create_app()
app.secret_key = 'dont tell anyone'
toastr = Toastr(app)

if __name__ == '__main__':
    from waitress import serve
    serve(app)
    # app.run(host='localhost', port=3000)
