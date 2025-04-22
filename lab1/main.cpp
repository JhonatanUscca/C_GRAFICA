//Include Libraries
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// Función que convierte una imagen en una matriz de valores RGB
std::vector<std::vector<std::vector<int>>> obtenerMatrizRGB(const cv::Mat& imagen) {
    int filas = imagen.rows;
    int columnas = imagen.cols;

    std::vector<std::vector<std::vector<int>>> matrizRGB(filas, std::vector<std::vector<int>>(columnas, std::vector<int>(3)));

    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < columnas; ++j) {
            cv::Vec3b pixel = imagen.at<cv::Vec3b>(i, j);

           
            matrizRGB[i][j][0] = pixel[2]; // R
            matrizRGB[i][j][1] = pixel[1]; // G
            matrizRGB[i][j][2] = pixel[0]; // B
        }
    }

    return matrizRGB;
}


// Función para gruardar los valores de una matriz a txt
void guardarMatrizRGBenTXT(const std::vector<std::vector<std::vector<int>>>& matrizRGB, const std::string& nombreArchivo) {
    std::ofstream archivo(nombreArchivo);

    if (!archivo.is_open()) {
        std::cerr << "No se pudo abrir el archivo para escribir." << std::endl;
        return;
    }

    int filas = matrizRGB.size();
    int columnas = matrizRGB[0].size();

    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < columnas; ++j) {
            archivo << "("
                << matrizRGB[i][j][0] << ","
                << matrizRGB[i][j][1] << ","
                << matrizRGB[i][j][2] << ") ";
        }
        archivo << "\n";
    }

    archivo.close();
    std::cout << "Matriz RGB guardada en: " << nombreArchivo << std::endl;
}


// Función para reconstruir una imagen desde un archivo de texto con RGB
cv::Mat reconstruirImagenDesdeTXT(const std::string& archivoTxt) {
    std::ifstream archivo(archivoTxt);
    if (!archivo.is_open()) {
        std::cerr << "No se pudo abrir el archivo para lectura." << std::endl;
        return cv::Mat();
    }

    std::string linea;
    std::vector<std::vector<cv::Vec3b>> datosRGB;

    while (std::getline(archivo, linea)) {
        std::stringstream ss(linea);
        std::string valor;
        std::vector<cv::Vec3b> fila;

        while (std::getline(ss, valor, ' ')) {
            if (valor.empty()) continue;

            valor.erase(std::remove(valor.begin(), valor.end(), '('), valor.end());
            valor.erase(std::remove(valor.begin(), valor.end(), ')'), valor.end());

            std::stringstream valorRGB(valor);
            std::string temp;
            int r, g, b;

            std::getline(valorRGB, temp, ','); r = std::stoi(temp);
            std::getline(valorRGB, temp, ','); g = std::stoi(temp);
            std::getline(valorRGB, temp, ','); b = std::stoi(temp);
            
            fila.push_back(cv::Vec3b(b, g, r));
        }

        datosRGB.push_back(fila);
    }

    archivo.close();

    // Crear imagen Mat
    int filas = datosRGB.size();
    int columnas = datosRGB[0].size();
    cv::Mat imagenReconstruida(filas, columnas, CV_8UC3);

    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < columnas; ++j) {
            imagenReconstruida.at<cv::Vec3b>(i, j) = datosRGB[i][j];
        }
    }

    return imagenReconstruida;
}


// Función que extrae canales R, G, B 
void extraerCanalesRGBenGrises(const std::string& archivoEntrada) {
    std::ifstream archivoIn(archivoEntrada);
    std::ofstream archivoRojo("canal_rojo.txt");
    std::ofstream archivoVerde("canal_verde.txt");
    std::ofstream archivoAzul("canal_azul.txt");

    if (!archivoIn.is_open() || !archivoRojo.is_open() || !archivoVerde.is_open() || !archivoAzul.is_open()) {
        std::cerr << " Error al abrir uno de los archivos." << std::endl;
        return;
    }

    std::string linea;
    while (std::getline(archivoIn, linea)) {
        std::stringstream ss(linea);
        std::string valor;

        while (std::getline(ss, valor, ' ')) {
            if (valor.empty()) continue;

            valor.erase(std::remove(valor.begin(), valor.end(), '('), valor.end());
            valor.erase(std::remove(valor.begin(), valor.end(), ')'), valor.end());

            std::stringstream valorRGB(valor);
            std::string temp;
            int r, g, b;

            std::getline(valorRGB, temp, ','); r = std::stoi(temp);
            std::getline(valorRGB, temp, ','); g = std::stoi(temp);
            std::getline(valorRGB, temp, ','); b = std::stoi(temp);

            archivoRojo << "(" << r << "," << r << "," << r << ") ";
            archivoVerde << "(" << g << "," << g << "," << g << ") ";
            archivoAzul << "(" << b << "," << b << "," << b << ") ";
        }
       
        archivoRojo << "\n";
        archivoVerde << "\n";
        archivoAzul << "\n";
    }
    archivoIn.close();
    archivoRojo.close();
    archivoVerde.close();
    archivoAzul.close();

}


int main() {
    // Cargar imagen
    //cv::Mat imagen = cv::imread("D:/grafica/a1.jpg");
    //cv::imshow("Imagen ", imagen);
    //cv::waitKey(0);


    // Obtener matriz RGB
    //auto matrizRGB = obtenerMatrizRGB(imagen);

    // Guardar matriz en archivo de texto
    //guardarMatrizRGBenTXT(matrizRGB, "matriz_rgb.txt");
    // Mostrar imagen original y reconstruida para comparar
    // 

    // Reconstruir imagen desde el archivo
    //cv::Mat imagenReconstruida = reconstruirImagenDesdeTXT("matriz_rgb.txt");
    //cv::imshow("Imagen Reconstruida", imagenReconstruida);
    //cv::waitKey(0);
    
    //extraerCanalesRGBenGrises("matriz_rgb.txt");
    cv::Mat imagenReconstruida = reconstruirImagenDesdeTXT("canal_rojo.txt");
    cv::imshow("Imagen Reconstruida", imagenReconstruida);
    cv::waitKey(0);

    return 0;
}
