STYLE = """
    QMainWindow {
        background-color: #F8F9FA;
    }

    QWidget {
        background-color: #F8F9FA;
        font-family: 'Segoe UI';
        font-size: 13px;
        color: #1A202C;
    }

    QFrame {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        border-radius: 6px;
    }

    QPushButton {
        background-color: #2B6CB0;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #2C5282;
    }

    QPushButton:pressed {
        background-color: #1A365D;
    }

    QLabel {
        background-color: transparent;
        border: none;
        color: #1A202C;
        font-size: 13px;
    }

    QLabel#label_titulo {
        font-size: 11px;
        font-weight: bold;
        color: #718096;
    }

    QLabel#label_valor {
        font-size: 13px;
        color: #1A202C;
    }

    QComboBox {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        border-radius: 5px;
        padding: 5px 10px;
        font-size: 13px;
        color: #1A202C;
    }

    QComboBox:hover {
        border: 1px solid #2B6CB0;
    }

    QComboBox:drop-down {
        border: none;
    }

    QSlider::groove:horizontal {
        height: 4px;
        background-color: #DEE2E6;
        border-radius: 2px;
    }

    QSlider::handle:horizontal {
        background-color: #2B6CB0;
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }

    QSlider::sub-page:horizontal {
        background-color: #2B6CB0;
        border-radius: 2px;
    }

    QCheckBox {
        font-size: 13px;
        color: #1A202C;
        background-color: transparent;
        border: none;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #DEE2E6;
        border-radius: 3px;
        background-color: #FFFFFF;
    }

    QCheckBox::indicator:checked {
        background-color: #2B6CB0;
        border: 1px solid #2B6CB0;
    }
"""