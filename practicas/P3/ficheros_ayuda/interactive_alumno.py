"""Completa los métodos pendientes de ComparisonSession y ejecuta launch_comparison
con encoder, dynamics y decoder preentrenados."""
from io import BytesIO
from pathlib import Path
import html
import time
import numpy as np
from PIL import Image
from carracing48 import make_env

ACTION_NAMES = {0: 'Nada', 1: 'Derecha', 2: 'Izquierda', 3: 'Acelerar', 4: 'Frenar'}

INITIAL_ACTIONS = (3, 0, 3, 1, 0, 3, 2, 0, 3, 0, 3)


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _load(model, custom_objects=None):
    if isinstance(model, (str, Path)):
        import keras
        model = keras.models.load_model(model, compile=False, custom_objects=custom_objects)
    if not callable(model):
        raise TypeError('Se espera un modelo Keras cargado o una ruta .keras.')
    def check(layer):
        if getattr(layer, 'stateful', False):
            raise ValueError('Usa stateful=False: el historial se proporciona en cada llamada.')
        for child in getattr(layer, 'layers', []):
            check(child)
    check(model)
    return model


def _frame(value):
    result = _array(value)
    if result.shape != (1,48,48,3):
        raise ValueError(f'El predictor debe devolver (1,48,48,3), no {result.shape}.')
    if not np.isfinite(result).all() or (result < -1e-6).any() or (result > 1+1e-6).any():
        raise ValueError('El predictor debe devolver imágenes finitas normalizadas a [0,1].')
    return np.clip(result[0],0,1).astype(np.float32)


class ComparisonSession:
    """Motor sin widgets; útil para reproducir partidas y comprobar ausencia de fugas."""
    def __init__(self, model=None, *, encoder=None, dynamics=None, decoder=None,
                 seed=42, custom_objects=None, env_factory=make_env):
        if model is not None:
            raise ValueError("Esta práctica requiere encoder, dynamics y decoder separados.")
        separate = (encoder, dynamics, decoder)
        if model is not None and any(m is not None for m in separate):
            raise ValueError('Pasa un predictor completo O los tres modelos separados.')
        if model is None and not all(m is not None for m in separate):
            raise ValueError('Falta el predictor completo o alguno de encoder/dynamics/decoder.')
        self.model = _load(model,custom_objects) if model is not None else None
        self.parts = tuple(_load(m,custom_objects) for m in separate) if model is None else None
        self.env_factory = env_factory
        self.env = None
        self.closed = False
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed < 0:
                raise ValueError('La semilla debe ser un entero no negativo.')
            self.seed = int(seed)
        if self.env is not None:
            self.env.close()
        self.env = self.env_factory()
        try:
            frame, _ = self.env.reset(seed=self.seed)

            for action in (3,)*15:
                frame, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    raise RuntimeError('El episodio terminó durante la inicialización.')
            frames = [frame.copy()]
            for action in INITIAL_ACTIONS:
                frame, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    raise RuntimeError('El episodio terminó durante el contexto inicial.')
                frames.append(frame.copy())
        except Exception:
            self.env.close()
            self.env = None
            self.closed = True
            raise
        self.context = np.stack(frames).astype(np.float32)/255.
        self.latent_context = None
        if self.parts is not None:
            try:
                self.latent_context = self._initialize_latents(self.context.copy())
            except Exception:
                self.env.close()
                self.closed = True
                raise
        self.history = np.array(INITIAL_ACTIONS,dtype=np.int64)
        self.real = frame.copy()
        self.generated = self.context[-1].copy()
        self.actions = []
        self.real_frames, self.generated_frames = [], []
        self.done = self.closed = False
        self.inference_ms = 0.
        return self.snapshot()

    def step(self, action):
        if self.closed or self.done:
            raise RuntimeError('Sesión cerrada o episodio terminado; reinicia para continuar.')
        if isinstance(action,(bool,np.bool_)) or not isinstance(action,(int,np.integer)) or int(action) not in ACTION_NAMES:
            raise ValueError('La acción debe ser un entero entre 0 y 4.')
        action = int(action)
        start = time.perf_counter()
        latent, prediction = self._predict_next(action)
        generated = _frame(prediction)
        elapsed = (time.perf_counter()-start)*1000

        real, _, terminated, truncated, _ = self.env.step(action)
        self._advance_history(latent, action)
        self.context = np.concatenate([self.context[1:],generated[None]])
        self.real, self.generated = real.copy(), generated
        self.done = bool(terminated or truncated)
        self.actions.append(action)
        self.real_frames.append(real.copy())
        self.generated_frames.append(generated.copy())
        self.inference_ms = elapsed
        return self.snapshot()



    def _initialize_latents(self, images):
        """Recibe (12,48,48,3) float [0,1]; devuelve (12,D) finito.

        Disponible: self.parts = (encoder, dynamics, decoder).
        Se llama una vez al iniciar y cada vez que se reinicia.
        """
        raise NotImplementedError("Completa la inicialización del estado del modelo")

    def _predict_next(self, action):
        """Devuelve (latente (1,D), imagen (1,48,48,3) en [0,1]).

        Disponibles: self.parts, self.latent_context (12,D),
        self.history (11,) y action (entero). No modifiques los historiales
        aquí: la salida debe validarse antes de avanzar el entorno real.
        No consultes self.real, self.env ni imágenes futuras.
        """
        raise NotImplementedError("Completa la predicción de un paso")

    def _advance_history(self, latent, action):
        """Actualiza self.latent_context y self.history conservando sus tamaños.

        Recibe la predicción validada y la acción aplicada a ambos sistemas.
        No vuelvas a codificar imágenes generadas ni uses observaciones reales.
        """
        raise NotImplementedError("Completa la actualización del estado temporal")

    def snapshot(self):
        return {'real': self.real.copy(), 'generated': self.generated.copy(),
                'step':len(self.actions), 'done':self.done, 'inference_ms':self.inference_ms,
                'actions':self.actions.copy(), 'seed':self.seed}

    def close(self):
        if self.env is not None:
            self.env.close()
        self.closed = True

    def trajectory(self):
        """Partida actual para análisis: imágenes normalizadas, sin contexto inicial."""
        if not self.actions:
            raise ValueError('Avanza al menos un paso antes de evaluar.')
        return {'real':np.stack(self.real_frames).astype(np.float32)/255.,
                'generated':np.stack(self.generated_frames),
                'actions':np.asarray(self.actions,dtype=np.int64)}


def _png(frame):
    if frame.dtype != np.uint8:
        frame = np.rint(frame*255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(frame).resize((288,288),Image.Resampling.NEAREST).save(output,format='PNG')
    return output.getvalue()




LIVE_ESM = r"""
export default {render({model, el}) {
  const pad = document.createElement('div');
  pad.tabIndex = 0;
  pad.style.cssText = 'padding:16px;border:2px solid #3679a8;border-radius:8px;outline-offset:3px';
  const button = document.createElement('button');
  const label = document.createElement('span');
  label.textContent = ' W acelerar · S frenar · A izquierda · D derecha · sin tecla: nada';
  const keyboard = document.createElement('input');
  keyboard.type='text'; keyboard.readOnly=true;
  keyboard.value='Control WASD: pulsa Jugar y mantén el foco aquí';
  keyboard.setAttribute('aria-label','Control del juego con WASD');
  keyboard.style.cssText='display:block;width:95%;margin-top:10px;padding:8px;cursor:default';
  const status = document.createElement('div');
  pad.append(button, label, status, keyboard); el.append(pad);
  const mapping = {KeyW:3, KeyS:4, KeyA:2, KeyD:1};
  const held = new Set();
  let running=false, pending=false, timer=null, token=null, sent=0, last=null;
  const period=1000/24;
  function stop(notify=true) {
    const old=token; running=false; token=null; held.clear(); clearTimeout(timer);
    button.textContent='Jugar (24 Hz)'; status.textContent='Pausado';
    if(notify && old) model.send({kind:'pause',token:old});
  }
  function tick() {
    if(!running || pending) return;
    const keys=Array.from(held);
    const action=keys.length ? mapping[keys[keys.length-1]] : 0;
    pending=true; sent=performance.now();
    model.send({kind:'tick',token,action});
  }
  function receive(msg) {
    if(msg.kind==='stopped') {pending=false; stop(false); return;}
    if(msg.token!==token) return;
    pending=false;
    if(msg.error || msg.done) {stop(false); status.textContent=msg.error || 'Fin del episodio'; return;}
    if(!running) return;
    const now=performance.now();
    if(last!==null) status.textContent=`Objetivo: 24 Hz · frecuencia efectiva: ${(1000/(now-last)).toFixed(1)} Hz`;
    last=now;
    timer=setTimeout(tick,Math.max(0,period-(now-sent)));
  }
  button.onclick=()=>{
    if(running) {stop(); return;}
    keyboard.focus(); held.clear(); running=true; pending=true; last=null;
    token=crypto.randomUUID(); sent=performance.now();
    button.textContent='Pausar'; status.textContent='Iniciando…';
    model.send({kind:'start',token});
  };
  function ownsKeyboard() {return running && pad.contains(document.activeElement);}
  function down(e) {
    if(!ownsKeyboard() || e.ctrlKey || e.metaKey || e.altKey) return;
    if(e.key==='Escape') {e.preventDefault(); e.stopImmediatePropagation(); stop(); return;}
    if(!(e.code in mapping)) return;
    e.preventDefault(); e.stopImmediatePropagation();
    if(!e.repeat) held.add(e.code);
  }
  function up(e) {
    if(!ownsKeyboard() || !(e.code in mapping)) return;
    e.preventDefault(); e.stopImmediatePropagation(); held.delete(e.code);
  }
  function blur(e) {if(!pad.contains(e.relatedTarget)) stop();}
  function hidden() {if(document.hidden) stop();}
  window.addEventListener('keydown',down,true);
  window.addEventListener('keyup',up,true);
  pad.addEventListener('focusout',blur);
  window.addEventListener('blur',stopOnBlur);
  function stopOnBlur(){stop();}
  document.addEventListener('visibilitychange',hidden);
  model.on('msg:custom',receive); stop(false);
  return ()=>{stop();model.off('msg:custom',receive);
    window.removeEventListener('blur',stopOnBlur);
    window.removeEventListener('keydown',down,true);
    window.removeEventListener('keyup',up,true);
    document.removeEventListener('visibilitychange',hidden);};
}};
"""

class ComparisonWidget:
    def __init__(self, session):
        import ipywidgets as w
        self.session = session
        self.busy = False
        try:
            import anywidget
        except ModuleNotFoundError as error:
            if error.name != 'anywidget':
                raise
            raise ModuleNotFoundError(
                'Falta anywidget en el kernel del notebook. Ejecuta en una celda: '
                '%pip install "anywidget>=0.9,<1" y vuelve a lanzar el comparador. '
                'Para instalar todas las dependencias: '
                '%pip install -r requirements-interactive.txt'
            ) from error
        class KeyboardControl(anywidget.AnyWidget):
            _esm = LIVE_ESM
        self.keyboard = KeyboardControl()
        self.keyboard.on_msg(self._live_message)
        self._live_token = None
        self.real = w.Image(format='png',width=288,height=288)
        self.generated = w.Image(format='png',width=288,height=288)
        self.status = w.HTML()
        self.metrics = w.HTML()
        self.seed = w.Text(value=str(session.seed),description='Semilla:')
        self.buttons = []
        for action in (2,3,1,4,0):
            button = w.Button(description=f'{action}: {ACTION_NAMES[action]}')
            button.on_click(lambda _,a=action:self.advance(a))
            self.buttons.append(button)
        self.reset_button = w.Button(description='Reiniciar')
        self.reset_button.on_click(lambda _: self.restart())
        self.close_button = w.Button(description='Cerrar')
        self.close_button.on_click(lambda _: self.close())
        self.record = w.Textarea(value='',description='Acciones:',disabled=True,
                                 layout=w.Layout(width='100%',height='65px'))
        self.replay_actions = w.Text(value='',description='Reproducir:',
                                     placeholder='Ejemplo: 3,3,1,0,4')
        self.replay_button = w.Button(description='Reproducir desde inicio')
        self.replay_button.on_click(lambda _:self.replay())
        self.evaluate_button = w.Button(description='Evaluar partida')
        self.evaluate_button.on_click(lambda _:self.evaluate())
        self.widget = w.VBox([
            w.HTML('<b>CarRacing: realidad y modelo</b><br>Flechas a 24 Hz objetivo; cada paso = 0,08 s simulados. '
                   'El modelo solo recibe sus propias imágenes después del contexto inicial.'),
            w.HBox([w.VBox([w.HTML('<b>Entorno real</b>'),self.real]),
                    w.VBox([w.HTML('<b>Modelo del alumno</b>'),self.generated])]),
            self.status, self.keyboard, w.HBox(self.buttons),
            w.HBox([self.seed,self.reset_button,self.close_button]),self.record,
            w.HBox([self.replay_actions,self.replay_button]),
            self.evaluate_button,self.metrics])
        self.refresh()

    def _stop_live(self):
        self._live_token = None
        self.keyboard.send({'kind':'stopped'})

    def _live_message(self, widget, message, buffers):
        kind, token = message.get('kind'), message.get('token')
        if kind == 'pause':
            if token == self._live_token:
                self._live_token = None
            return
        if kind == 'start':
            if self.session.closed or self.session.done or self.busy:
                self.keyboard.send({'token':token, 'done':True})
                return
            self._live_token = token
            self.keyboard.send({'token':token, 'done':False})
            return
        if kind != 'tick' or not token or token != self._live_token:
            return
        try:
            self.session.step(message.get('action'))
            self.refresh()
            self.keyboard.send({'token':token, 'done':self.session.done})
            if self.session.done:
                self._live_token = None
        except Exception as error:
            self._live_token = None
            self.status.value = '<b>Error:</b> '+html.escape(str(error))
            self.keyboard.send({'token':token, 'error':str(error)})

    def refresh(self):
        state = self.session.snapshot()
        self.real.value, self.generated.value = _png(state['real']),_png(state['generated'])
        end = ' · Fin del episodio: reinicia.' if state['done'] else ''
        self.status.value = (f"Paso {state['step']} · tiempo simulado {state['step']*.08:.2f} s"
                             f" · inferencia {state['inference_ms']:.1f} ms{end}")
        self.record.value = ','.join(map(str,state['actions']))
        self.metrics.value = ''
        for button in self.buttons:
            button.disabled = state['done'] or self.session.closed

    def _run(self, operation):
        if self.busy:
            return
        self.busy = True
        try:
            operation()
            self.refresh()
        except Exception as error:
            self.status.value = '<b>Error:</b> '+html.escape(str(error))
        finally:
            self.busy = False

    def advance(self, action):
        self._stop_live()
        self._run(lambda:self.session.step(action))

    def restart(self):
        self._stop_live()
        self._run(lambda:self.session.reset(int(self.seed.value)))

    def replay(self):
        self._stop_live()
        def run():
            actions = [int(s.strip()) for s in self.replay_actions.value.split(',') if s.strip()]
            if not actions or any(a not in ACTION_NAMES for a in actions) or len(actions)>224:
                raise ValueError('Introduce entre 1 y 224 acciones, enteros 0..4 separados por comas.')
            self.session.reset(int(self.seed.value))
            for action in actions:
                self.session.step(action)
                self.refresh()
                if self.session.done:
                    break
        self._run(run)

    def close(self):
        self._stop_live()
        self.session.close()
        for button in self.buttons:
            button.disabled = True
        self.status.value = 'Sesión cerrada. Reinicia para volver a jugar.'

    def evaluate(self):
        self._stop_live()
        from metric import visual_metrics
        try:
            data = self.session.trajectory()
            result = visual_metrics(data['real'],data['generated'])
            self.metrics.value = ('<b>Partida actual:</b> ' + ' · '.join(
                f'{name}: {result[name]:.5f}' for name in ('score','ssim','gradient','mse')))
        except Exception as error:
            self.metrics.value = '<b>Error:</b> '+html.escape(str(error))


def launch_comparison(model=None, *, encoder=None, dynamics=None, decoder=None,
                      seed=42, custom_objects=None):
    """Muestra los dos paneles y devuelve la aplicación. No entrena modelos."""
    from IPython.display import display
    session = ComparisonSession(model,encoder=encoder,dynamics=dynamics,decoder=decoder,
                                seed=seed,custom_objects=custom_objects)
    try:
        app = ComparisonWidget(session)
    except Exception:
        session.close()
        raise
    display(app.widget)
    return app
