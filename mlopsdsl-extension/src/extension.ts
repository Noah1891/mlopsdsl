import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as cp from 'child_process';
import { ParameterizedLanguageServer, VSCodeUriResolverServer, LanguageParameter } from '@usethesource/rascal-vscode-dsl-lsp-server';

/** Name des virtuellen Environments - identisch zum Namen, der im
 *  Entwicklungsmodus unter `src/main/python/.mlopsenv` liegt. Dadurch bleibt
 *  der von Rascal aus per `findResources` gesuchte relative Pfad
 *  (".mlopsenv/bin/python3" bzw. ".mlopsenv/Scripts/python.exe") in beiden
 *  Modi identisch - nur die Wurzel, die zur pathConfig hinzugefügt wird,
 *  unterscheidet sich. */
const VENV_FOLDER_NAME = '.mlopsenv';

export async function activate(context: vscode.ExtensionContext) {
	context.subscriptions.push(
		vscode.commands.registerCommand('mlopsdsl.setupPythonEnvironment', () =>
			setupPythonEnvironment(context, true))
	);

	// Prüfen/Anbieten läuft im Hintergrund, blockiert die Aktivierung nicht.
	void ensurePythonEnvironment(context);

	// jar, das das "Plugin" Modul (die eigentliche Sprachdefinition) enthält
	const mlopsLSPJar = `|jar+file://${context.extensionUri.path}/assets/jars/mlopsdsl-lsp.jar!|`;
	// die Python-Skripte liegen NICHT im JAR (createProcess kann keine
	// jar+file:// URIs an einen externen Prozess übergeben), sondern als
	// echte Dateien im Extension-Verzeichnis - siehe assets/python
	const pythonScriptsLoc = `|file://${context.extensionUri.path}/assets/python|`;
	// persistenter, System- und Projekt-unabhängiger Ort, unter dem die
	// Extension das virtuelle Environment nach der Installation anlegt
	const venvRootLoc = `|file://${venvRootUriPath(context)}|`;

	const language = <LanguageParameter>{
		pathConfig: `pathConfig(srcs=[${mlopsLSPJar}, ${pythonScriptsLoc}, ${venvRootLoc}])`,
		name: "MLOps DSL",
		extensions: ["mlops"],
		mainModule: "Plugin",
		mainFunction: "pipelineLanguageServices"
	};

	// rascal vscode braucht eine Instanz dieser Klasse; bei mehreren Sprachen
	// kann sie gemeinsam genutzt werden
	const vfs = new VSCodeUriResolverServer(false);
	// startet den LSP-Server und verbindet ihn mit Rascal
	const lsp = new ParameterizedLanguageServer(context,
		vfs,
		calcJarPath(context),
		true,
		"mlops",     // vscode language ID
		"MLOps DSL", // vscode language Titel (unten rechts sichtbar)
		language);
	// über subscriptions wird beim Deaktivieren alles korrekt aufgeräumt
	context.subscriptions.push(lsp);
}

function calcJarPath(context: vscode.ExtensionContext) {
	return context.asAbsolutePath(path.join('.', 'dist', 'rascal-lsp'));
}

/** Native Dateisystem-Variante des persistenten Speicherorts (für fs/child_process). */
function venvRootFsPath(context: vscode.ExtensionContext): string {
	// globalStorageUri ist pro Extension stabil - unabhängig vom Projekt/
	// Workspace und vom Rechner/Benutzernamen der Person, die die Extension
	// installiert. Genau deshalb ist es der richtige, portable Ort für ein
	// einmalig angelegtes virtuelles Environment.
	return context.globalStorageUri.fsPath;
}

/** URI-Pfad-Variante desselben Ortes (für die Rascal `pathConfig`-Location). */
function venvRootUriPath(context: vscode.ExtensionContext): string {
	return context.globalStorageUri.path;
}

function venvDir(context: vscode.ExtensionContext): string {
	return path.join(venvRootFsPath(context), VENV_FOLDER_NAME);
}

function pythonExecutablePath(venv: string): string {
	return process.platform === 'win32'
		? path.join(venv, 'Scripts', 'python.exe')
		: path.join(venv, 'bin', 'python3');
}

async function ensurePythonEnvironment(context: vscode.ExtensionContext) {
	const venv = venvDir(context);
	if (fs.existsSync(pythonExecutablePath(venv))) {
		return; // bereits eingerichtet
	}

	const choice = await vscode.window.showInformationMessage(
		'Die MLOps DSL benötigt eine lokale Python-Umgebung (für Schema-Inferenz, Typprüfung und Pipeline-Ausführung). Jetzt einrichten?',
		'Jetzt einrichten', 'Später'
	);
	if (choice === 'Jetzt einrichten') {
		await setupPythonEnvironment(context, false);
	}
}

/** Signalisiert, dass der konfigurierte Python-Befehl gar nicht existiert
 *  (z.B. weil auf dem System kein Python installiert ist), im Unterschied
 *  zu einem Fehler *während* der Ausführung von venv/pip. */
class PythonNotFoundError extends Error {}

async function setupPythonEnvironment(context: vscode.ExtensionContext, showSuccessMessage: boolean) {
	const venv = venvDir(context);
	const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
	const requirements = context.asAbsolutePath(path.join('assets', 'python', 'requirements.txt'));

	await vscode.window.withProgress({
		location: vscode.ProgressLocation.Notification,
		title: 'MLOps DSL: Python-Umgebung wird eingerichtet…',
		cancellable: false
	}, async (progress) => {
		try {
			fs.mkdirSync(venvRootFsPath(context), { recursive: true });

			progress.report({ message: 'Erzeuge virtuelles Environment…' });
			await run(pythonCmd, ['-m', 'venv', venv]);

			const pip = pythonExecutablePath(venv);
			progress.report({ message: 'Aktualisiere pip…' });
			await run(pip, ['-m', 'pip', 'install', '--upgrade', 'pip']);

			if (fs.existsSync(requirements)) {
				progress.report({ message: 'Installiere Abhängigkeiten…' });
				await run(pip, ['-m', 'pip', 'install', '-r', requirements]);
			}

			if (showSuccessMessage) {
				void vscode.window.showInformationMessage('MLOps DSL: Python-Umgebung erfolgreich eingerichtet.');
			}
		} catch (err) {
			if (err instanceof PythonNotFoundError) {
				await handlePythonNotFound(context, pythonCmd, showSuccessMessage);
			} else {
				void vscode.window.showErrorMessage(`MLOps DSL: Einrichtung der Python-Umgebung fehlgeschlagen: ${err}`);
			}
		}
	});
}

async function handlePythonNotFound(context: vscode.ExtensionContext, pythonCmd: string, showSuccessMessage: boolean) {
	const choice = await vscode.window.showErrorMessage(
		`MLOps DSL: Es wurde kein "${pythonCmd}" gefunden. Bitte installiere Python 3, danach kannst du es hier erneut versuchen.`,
		'Python-Downloadseite öffnen', 'Erneut versuchen'
	);

	if (choice === 'Python-Downloadseite öffnen') {
		void vscode.env.openExternal(vscode.Uri.parse('https://www.python.org/downloads/'));
		// direkt einen Retry anbieten, statt den Nutzer auf den Befehl in der
		// Befehlspalette zu verweisen - er muss den Namen dafür nicht kennen
		const retry = await vscode.window.showInformationMessage(
			'MLOps DSL: Sobald die Python-Installation abgeschlossen ist, kannst du die Einrichtung hier erneut versuchen.',
			'Erneut versuchen'
		);
		if (retry === 'Erneut versuchen') {
			await setupPythonEnvironment(context, showSuccessMessage);
		}
	} else if (choice === 'Erneut versuchen') {
		await setupPythonEnvironment(context, showSuccessMessage);
	}
}

function run(cmd: string, args: string[]): Promise<void> {
	return new Promise((resolve, reject) => {
		const proc = cp.spawn(cmd, args, { shell: process.platform === 'win32' });
		let stderr = '';
		proc.stderr?.on('data', (d) => { stderr += d.toString(); });
		proc.on('error', (err: NodeJS.ErrnoException) => {
			if (err.code === 'ENOENT') {
				reject(new PythonNotFoundError(`Befehl "${cmd}" wurde nicht gefunden.`));
			} else {
				reject(err);
			}
		});
		proc.on('close', (code: number | null) => {
			if (code === 0) {
				resolve();
			} else {
				reject(new Error(`${cmd} ${args.join(' ')} beendete sich mit Code ${code}: ${stderr}`));
			}
		});
	});
}

export function deactivate() {}