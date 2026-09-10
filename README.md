# Detectar la mano delante de la pantalla con efecto Doppler

Un amigo me mandó un vídeo de algo parecido funcionando: un portátil normal, sin
hardware añadido, detectando la mano con el altavoz y el micrófono. Mi reacción
fue **"no hay forma de que eso funcione"**.

Así que lo vibecodeé una tarde para quitarme la espina, sin creérmelo en ningún
momento y sin la menor intención de que esto llegara a ninguna parte. Para mi
sorpresa, funciona.

![demo](demo.gif)

## Los fundamentos, en corto

El altavoz emite un tono continuo a ~20 kHz, inaudible para casi cualquier
adulto. El micrófono lo capta por camino directo —la **portadora**, fortísima y
perfectamente estable— más los ecos de la habitación. Todo lo que está quieto
refleja en la misma frecuencia. Lo que se mueve devuelve el eco desplazado:

$$\Delta f = \frac{2vf_0}{c}$$

Una mano a 30 cm/s sobre 20 kHz da unos 35 Hz. En relativo es un 0.17 %, una
miseria; pero al lado de una raya espectral tan limpia como la portadora, son
seis bins de la FFT y se ven perfectamente.

El programa mide la energía que aparece a los lados de la portadora, la divide
por la potencia de la portadora y compara el resultado con la línea base de la
habitación vacía. Esa división es lo que hace que el invento sobreviva al
control automático de ganancia del micro: si el sistema sube o baja el volumen
de entrada, portadora y bandas laterales suben y bajan juntas, y el cociente ni
se entera.

Que el eco venga por arriba o por abajo en frecuencia dice si te acercas o te
alejas. Eso es toda la física que hay aquí.

## Cómo hacerlo funcionar en tu ordenador

```bash
git clone https://github.com/malmriv/doppler.git
cd doppler
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python doppler_hand.py --auto-freq
```

Tarda unos 15 segundos en arrancar: busca la mejor portadora, espera a que el
micrófono se estabilice y calibra la habitación. **Aparta las manos durante la
calibración.** Después, mueve la mano delante de la pantalla. `Ctrl-C` para
salir.

Ahora las condiciones, que importan más que el código:

- **Altavoces internos, con volumen.** Al 50 % va sobrado. Si los tienes
  silenciados no hay nada que detectar.
- **Nada de Bluetooth ni auriculares.** Los AirPods y similares ni llegan a
  20 kHz ni transmiten sin comprimir, y con auriculares puestos el tono no sale
  al aire. Altavoz y micro **internos**, los dos.
- **El micrófono tiene que estar quieto.** Si mueves el portátil, se mueve la
  geometría entera y todo lo demás parece moverse contigo. En el regazo no
  funciona; en una mesa, sí.
- **Concede el permiso de micrófono** cuando lo pida el sistema.
- **La habitación no tiene por qué estar en silencio.** El ruido normal (voces,
  teclado, ventiladores) vive por debajo de 8 kHz; a 20 kHz hay 37 dB menos de
  ruido. Es el rincón más tranquilo del espectro, y por eso el tono es
  ultrasónico y no audible.
- **Cuidado con quién más se mueve por la habitación.** Un tono continuo no mide
  distancia: no distingue tu mano de alguien que pase por detrás.

Probado en un MacBook Pro. En Linux y Windows debería ir igual (`sounddevice`
usa PortAudio en los tres sitios), pero no lo he comprobado.

### Si no va

| Síntoma | Causa habitual |
|---|---|
| Avisa de que "apenas se recibe la portadora" | Volumen bajo, salida por Bluetooth, o auriculares puestos |
| No detecta nada aunque muevas la mano | Calibraste con movimiento delante; reinícialo y aparta las manos |
| Detecta constantemente | Algo se mueve cerca: un ventilador, una cortina, alguien pasando |
| Nada de nada, ni portadora ni ruido | Falta el permiso de micrófono |

Opciones útiles: `--freq 17000` si tu hardware no llega arriba, `--margin` para
hacerlo más o menos sensible, `--csv fichero.csv` para volcar las medidas y
mirarlas con calma.

## Qué se ve en pantalla

```
  señal     [██████████░░░░░░░░░░░░]  -28.9 dB   base -42.1  umbral -35.5
  veloc.  ◄···········█████│················►  -0.29 m/s
  historia  ····▁▁▁······▁▁▂▂▃▃▄▄▅▅▅▄▄▃▃▂▂▂▁·▁▁▂▂▂
  estado    ● MANO  alejándose    portadora -0.1 dB   20000 Hz
```

Verde y hacia la derecha, acercándose; cian y hacia la izquierda, alejándose. La
fila de historia son los últimos segundo y pico.

## ¿Pero funciona de verdad?

Eso me preguntaba yo. Medido en un MacBook Pro:

- La portadora se recibe a **-19 dBFS** a 21 kHz. Holgadísima.
- En calma la línea base es de **-42 dB** con una desviación robusta de 2.4 dB,
  y el umbral queda 6-8 dB por encima. El detector responde a ecos de hasta
  **-40 dB** respecto a la portadora.
- En 25 segundos de habitación "vacía" salieron cuatro episodios limpios de
  entre medio segundo y segundo y medio, a unos 0.43 m/s. No eran falsos
  positivos: era yo, moviéndome.

## Lo que no hace

El Doppler detecta **movimiento, no presencia**. Una mano perfectamente inmóvil
delante de la pantalla no genera bandas laterales y el programa no la ve. Sí
altera un poco la amplitud de la portadora, y eso se muestra como métrica
secundaria, pero es bastante menos fiable.

Tampoco mide distancia, con lo cual no puede separar tu mano de lo que ocurra al
fondo de la habitación. Para eso haría falta un chirp FMCW en vez de un tono
fijo, que da distancia y velocidad a la vez. Ahí ya no llegué: esto era una
tarde para demostrarme que no funcionaba.

## Licencia

MIT. Haz lo que quieras con ello.
