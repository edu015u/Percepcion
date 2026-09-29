#include <Arduino_RouterBridge.h>

const int pinVerde = 8;
const int pinNaranja = 9;
const int pinRojo = 10;

// Estados: 0 = apagados, 1 = verde, 2 = naranja, 3 = rojo.
const unsigned long tiempoSinOrden = 5000;
unsigned long ultimaOrden = 0;
int estadoActual = 0;

void apagarTodos() {
  digitalWrite(pinVerde, LOW);
  digitalWrite(pinNaranja, LOW);
  digitalWrite(pinRojo, LOW);
}

int establecerEstado(int estado) {
  if (estado < 0 || estado > 3) {
    return -1;
  }

  apagarTodos();

  if (estado == 1) digitalWrite(pinVerde, HIGH);
  if (estado == 2) digitalWrite(pinNaranja, HIGH);
  if (estado == 3) digitalWrite(pinRojo, HIGH);

  estadoActual = estado;
  ultimaOrden = millis();
  return estado;  // Confirmacion para Python, despues de actualizar los pines.
}

void setup() {
  pinMode(pinVerde, OUTPUT);
  pinMode(pinNaranja, OUTPUT);
  pinMode(pinRojo, OUTPUT);
  apagarTodos();

  Bridge.begin();
  Bridge.provide_safe("establecer_estado", establecerEstado);
}

void loop() {
  // Apagar si Python deja de enviar ordenes durante 5 segundos.
  if (estadoActual != 0 && millis() - ultimaOrden >= tiempoSinOrden) {
    apagarTodos();
    estadoActual = 0;
  }
  delay(1);
}
