#ifndef DLART_GUI_H
#define DLART_GUI_H

#include <QMainWindow>

namespace Ui {
class DLArt_GUI;
}

class DLArt_GUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit DLArt_GUI(QWidget *parent = 0);
    ~DLArt_GUI();

    int getPatchSizeX();
    int getPatchSizeY();
    float getPatchOverlapp();

private slots:
    void on_Button_Patching_clicked();

    void on_Button_DB_clicked();

    void on_TreeWidget_Patients_clicked(const QModelIndex &index);

    void patientSelectionChanged();

private:
    Ui::DLArt_GUI *ui;

    QString databasePath;

    //Attributes
    int m_patchSizeX;
    int m_patchSizeY;
    float m_patchOverlapp;


};

#endif // DLART_GUI_H
