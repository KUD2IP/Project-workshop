from flask import Flask, request, render_template, send_file, redirect, url_for
import os
import zipfile
import torch
from torchvision import datasets, transforms, models
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
ALLOWED_EXTENSIONS = {'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_data(upload_folder):
    # Здесь мы можем распаковать архив и подготовить данные для обучения
    for filename in os.listdir(upload_folder):
        if filename.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(upload_folder, filename), 'r') as zip_ref:
                zip_ref.extractall(upload_folder)
            os.remove(os.path.join(upload_folder, filename))  # Удаляем архив после распаковки


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        prepare_data(app.config['UPLOAD_FOLDER'])
        return redirect(url_for('train_model'))

    return "Invalid file format"


@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        epochs = int(request.form.get('epochs', 5))

        # Пример подготовки данных и обучения модели
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        train_data = datasets.ImageFolder(UPLOAD_FOLDER, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(train_data.classes))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        # Сохранение весов модели
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        model_path = os.path.join(MODEL_FOLDER, 'model.pth')
        torch.save(model.state_dict(), model_path)

        return redirect(url_for('download_weights'))

    return render_template('train.html')


@app.route('/download')
def download_weights():
    try:
        return send_file(os.path.join(MODEL_FOLDER, 'model.pth'), as_attachment=True)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
