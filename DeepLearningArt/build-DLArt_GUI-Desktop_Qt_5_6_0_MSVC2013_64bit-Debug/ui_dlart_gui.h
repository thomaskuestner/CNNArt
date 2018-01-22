/********************************************************************************
** Form generated from reading UI file 'dlart_gui.ui'
**
** Created by: Qt User Interface Compiler version 5.6.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DLART_GUI_H
#define UI_DLART_GUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLArt_GUI
{
public:
    QWidget *centralWidget;
    QTabWidget *tabWidget;
    QWidget *Tab_Multiclass;
    QGroupBox *groupBox;
    QSpinBox *SpinBox_PatchX;
    QLabel *label_5;
    QLabel *label_4;
    QSpinBox *SpinBox_PatchY;
    QLabel *label_6;
    QDoubleSpinBox *SpinBox_PatchOverlapp;
    QPushButton *Button_Patching;
    QTreeWidget *TreeWidget_Patients;
    QTreeWidget *TreeWidget_Datasets;
    QPushButton *Button_DB;
    QLabel *Label_DB;
    QPushButton *Button_OutputPathPatching;
    QLabel *Label_OutputPathPatching;
    QPushButton *pushButton;
    QLabel *label;
    QRadioButton *radioButton;
    QRadioButton *radioButton_2;
    QGraphicsView *GraphicsView_Pic;
    QGroupBox *groupBox_2;
    QGroupBox *groupBox_3;
    QWidget *Tab_GAN;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *DLArt_GUI)
    {
        if (DLArt_GUI->objectName().isEmpty())
            DLArt_GUI->setObjectName(QStringLiteral("DLArt_GUI"));
        DLArt_GUI->resize(1385, 904);
        centralWidget = new QWidget(DLArt_GUI);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(10, 10, 1361, 811));
        Tab_Multiclass = new QWidget();
        Tab_Multiclass->setObjectName(QStringLiteral("Tab_Multiclass"));
        groupBox = new QGroupBox(Tab_Multiclass);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 511, 431));
        SpinBox_PatchX = new QSpinBox(groupBox);
        SpinBox_PatchX->setObjectName(QStringLiteral("SpinBox_PatchX"));
        SpinBox_PatchX->setGeometry(QRect(100, 241, 60, 31));
        QFont font;
        font.setPointSize(10);
        SpinBox_PatchX->setFont(font);
        SpinBox_PatchX->setMaximum(500);
        SpinBox_PatchX->setValue(40);
        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(170, 240, 16, 31));
        QFont font1;
        font1.setPointSize(14);
        label_5->setFont(font1);
        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(20, 240, 71, 31));
        label_4->setFont(font);
        SpinBox_PatchY = new QSpinBox(groupBox);
        SpinBox_PatchY->setObjectName(QStringLiteral("SpinBox_PatchY"));
        SpinBox_PatchY->setGeometry(QRect(190, 240, 60, 31));
        SpinBox_PatchY->setFont(font);
        SpinBox_PatchY->setMaximum(500);
        SpinBox_PatchY->setValue(40);
        label_6 = new QLabel(groupBox);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(340, 240, 61, 31));
        label_6->setFont(font);
        SpinBox_PatchOverlapp = new QDoubleSpinBox(groupBox);
        SpinBox_PatchOverlapp->setObjectName(QStringLiteral("SpinBox_PatchOverlapp"));
        SpinBox_PatchOverlapp->setGeometry(QRect(400, 240, 71, 31));
        SpinBox_PatchOverlapp->setFont(font);
        SpinBox_PatchOverlapp->setMaximum(1);
        SpinBox_PatchOverlapp->setSingleStep(0.05);
        SpinBox_PatchOverlapp->setValue(0.5);
        Button_Patching = new QPushButton(groupBox);
        Button_Patching->setObjectName(QStringLiteral("Button_Patching"));
        Button_Patching->setGeometry(QRect(410, 380, 91, 28));
        Button_Patching->setFont(font);
        TreeWidget_Patients = new QTreeWidget(groupBox);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(0, QStringLiteral("1"));
        TreeWidget_Patients->setHeaderItem(__qtreewidgetitem);
        TreeWidget_Patients->setObjectName(QStringLiteral("TreeWidget_Patients"));
        TreeWidget_Patients->setGeometry(QRect(10, 78, 151, 151));
        TreeWidget_Datasets = new QTreeWidget(groupBox);
        QTreeWidgetItem *__qtreewidgetitem1 = new QTreeWidgetItem();
        __qtreewidgetitem1->setText(0, QStringLiteral("1"));
        TreeWidget_Datasets->setHeaderItem(__qtreewidgetitem1);
        TreeWidget_Datasets->setObjectName(QStringLiteral("TreeWidget_Datasets"));
        TreeWidget_Datasets->setGeometry(QRect(180, 78, 321, 151));
        Button_DB = new QPushButton(groupBox);
        Button_DB->setObjectName(QStringLiteral("Button_DB"));
        Button_DB->setGeometry(QRect(10, 30, 111, 31));
        Button_DB->setFont(font);
        Label_DB = new QLabel(groupBox);
        Label_DB->setObjectName(QStringLiteral("Label_DB"));
        Label_DB->setGeometry(QRect(130, 35, 361, 21));
        QFont font2;
        font2.setPointSize(8);
        Label_DB->setFont(font2);
        Button_OutputPathPatching = new QPushButton(groupBox);
        Button_OutputPathPatching->setObjectName(QStringLiteral("Button_OutputPathPatching"));
        Button_OutputPathPatching->setGeometry(QRect(20, 380, 91, 31));
        Button_OutputPathPatching->setFont(font);
        Label_OutputPathPatching = new QLabel(groupBox);
        Label_OutputPathPatching->setObjectName(QStringLiteral("Label_OutputPathPatching"));
        Label_OutputPathPatching->setGeometry(QRect(120, 380, 271, 31));
        Label_OutputPathPatching->setFont(font2);
        pushButton = new QPushButton(groupBox);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(20, 335, 91, 31));
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(120, 340, 81, 21));
        radioButton = new QRadioButton(groupBox);
        radioButton->setObjectName(QStringLiteral("radioButton"));
        radioButton->setGeometry(QRect(40, 300, 121, 17));
        radioButton->setFont(font);
        radioButton_2 = new QRadioButton(groupBox);
        radioButton_2->setObjectName(QStringLiteral("radioButton_2"));
        radioButton_2->setGeometry(QRect(200, 300, 101, 17));
        radioButton_2->setFont(font);
        GraphicsView_Pic = new QGraphicsView(Tab_Multiclass);
        GraphicsView_Pic->setObjectName(QStringLiteral("GraphicsView_Pic"));
        GraphicsView_Pic->setGeometry(QRect(540, 360, 451, 391));
        groupBox_2 = new QGroupBox(Tab_Multiclass);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 460, 511, 311));
        groupBox_3 = new QGroupBox(Tab_Multiclass);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setGeometry(QRect(540, 20, 451, 331));
        tabWidget->addTab(Tab_Multiclass, QString());
        Tab_GAN = new QWidget();
        Tab_GAN->setObjectName(QStringLiteral("Tab_GAN"));
        tabWidget->addTab(Tab_GAN, QString());
        DLArt_GUI->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(DLArt_GUI);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1385, 21));
        DLArt_GUI->setMenuBar(menuBar);
        mainToolBar = new QToolBar(DLArt_GUI);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        DLArt_GUI->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(DLArt_GUI);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        DLArt_GUI->setStatusBar(statusBar);

        retranslateUi(DLArt_GUI);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(DLArt_GUI);
    } // setupUi

    void retranslateUi(QMainWindow *DLArt_GUI)
    {
        DLArt_GUI->setWindowTitle(QApplication::translate("DLArt_GUI", "Deep Learning Art", 0));
        groupBox->setTitle(QApplication::translate("DLArt_GUI", "Database and Patching", 0));
        label_5->setText(QApplication::translate("DLArt_GUI", "x", 0));
        label_4->setText(QApplication::translate("DLArt_GUI", "Patch Size:", 0));
        label_6->setText(QApplication::translate("DLArt_GUI", "Overlap:", 0));
        Button_Patching->setText(QApplication::translate("DLArt_GUI", "Patching", 0));
        Button_DB->setText(QApplication::translate("DLArt_GUI", "Select Database", 0));
        Label_DB->setText(QApplication::translate("DLArt_GUI", "No Database selected", 0));
        Button_OutputPathPatching->setText(QApplication::translate("DLArt_GUI", "Output Path", 0));
        Label_OutputPathPatching->setText(QApplication::translate("DLArt_GUI", "Output Path", 0));
        pushButton->setText(QApplication::translate("DLArt_GUI", "Markings Path", 0));
        label->setText(QApplication::translate("DLArt_GUI", "Markings Path", 0));
        radioButton->setText(QApplication::translate("DLArt_GUI", "Mask Labeling", 0));
        radioButton_2->setText(QApplication::translate("DLArt_GUI", "Patch Labeling", 0));
        groupBox_2->setTitle(QApplication::translate("DLArt_GUI", "Deep Neural Network", 0));
        groupBox_3->setTitle(QApplication::translate("DLArt_GUI", "Training Parameters", 0));
        tabWidget->setTabText(tabWidget->indexOf(Tab_Multiclass), QApplication::translate("DLArt_GUI", "Multiclass Classification", 0));
        tabWidget->setTabText(tabWidget->indexOf(Tab_GAN), QApplication::translate("DLArt_GUI", "ArtGAN", 0));
    } // retranslateUi

};

namespace Ui {
    class DLArt_GUI: public Ui_DLArt_GUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DLART_GUI_H
