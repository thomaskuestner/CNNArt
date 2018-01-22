#include "dlart_gui.h"
#include "ui_dlart_gui.h"
#include <QFileDialog>
#include <QFileSystemModel>
#include <QStandardItemModel>
#include <QFontDatabase>
#include <QDebug>

DLArt_GUI::DLArt_GUI(QWidget *parent) :
    QMainWindow(parent),
    databasePath("D:/med_data/MRPhysics/newProtocol"),
    ui(new Ui::DLArt_GUI)
{
    ui->setupUi(this);

    ui->Label_DB->setText(databasePath);

    connect(ui->TreeWidget_Patients, SIGNAL(itemSelectionChanged()), this, SLOT(patientSelectionChanged()));
}

DLArt_GUI::~DLArt_GUI()
{
    delete ui;
}

void DLArt_GUI::on_Button_Patching_clicked()
{

}

void DLArt_GUI::patientSelectionChanged()
{
    QList<QTreeWidgetItem*> selectedItems = ui->TreeWidget_Patients->selectedItems();

    //ui->Label_Overview->setText(QString::number(selectedItems.size()));
    //qDebug << "Selection Changed" << QString::number(selectedItems.size());
}

void DLArt_GUI::on_Button_DB_clicked()
{
    databasePath = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "D:/med_data/MRPhysics/newProtocol",
            QFileDialog::ShowDirsOnly|QFileDialog::DontResolveSymlinks);


    ui->Label_DB->setText(databasePath);

    QFileSystemModel *model2 = new QFileSystemModel;
    model2->setRootPath(databasePath);

    //Tree View
    QFontDatabase database;

    ui->TreeWidget_Patients->setColumnCount(1);
    ui->TreeWidget_Patients->setHeaderLabels(QStringList() <<tr("Patients"));

    foreach(QString family, database.families()){
        const QStringList styles = database.styles(family);
        if(styles.isEmpty())
            continue;

        QTreeWidgetItem *familyItem = new QTreeWidgetItem(ui->TreeWidget_Patients);
        familyItem->setText(0, family);
        familyItem->setCheckState(0, Qt::Unchecked);
        familyItem->setFlags(familyItem->flags() | Qt::ItemIsAutoTristate);

        foreach(QString style, styles)        {
            QTreeWidgetItem *styleItem = new QTreeWidgetItem(familyItem);
            styleItem->setText(0, style);
            styleItem->setCheckState(0, Qt::Unchecked);
            styleItem->setData(0, Qt::UserRole, QVariant(database.weight(family, style)));
            styleItem->setData(0, Qt::UserRole+1, QVariant(database.italic(family, style)));
        }
    }

    //Table
    /*QStandardItemModel *model = new QStandardItemModel(10, 2, this);
    model->setHorizontalHeaderItem(0, new QStandardItem(QString("Patient")));
    model->setHorizontalHeaderItem(1, new QStandardItem(QString("Used")));

    QStandardItem *firstRow = new QStandardItem(QString("Column Value"));
    model->setItem(0,1,firstRow);

    ui->Table_Patients->setModel(model);

    */

    //ui->Table_Patients->setModel();
}

void DLArt_GUI::on_TreeWidget_Patients_clicked(const QModelIndex &index)
{
    QList<QTreeWidgetItem*> selectedItems = ui->TreeWidget_Patients->selectedItems();
    QString rets="";
    //ui->Label_Overview->setText(QString::number(selectedItems.size()));
    //ui->Label_Overview->
    //for(int i=0; selectedItems.size(); i++)
        //(rets = rets + selectedItems.at(i)->text(0);

}




