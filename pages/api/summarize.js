import { spawn } from 'child_process';
import path from 'path';
import { promisify } from 'util';
import fs from 'fs';

const writeFile = promisify(fs.writeFile);

const summarizeText = async (text) => {
  const scriptPath = path.join(process.cwd(), 'scripts', 'summarize.py');
  const result = await new Promise((resolve, reject) => {
    const pythonProcess = spawn('python3', [scriptPath, text]);

    let data = '';
    pythonProcess.stdout.on('data', (chunk) => {
      data += chunk.toString();
    });

    pythonProcess.stderr.on('data', (error) => {
      reject(error.toString());
    });

    pythonProcess.on('close', () => {
      resolve(data.trim());
    });
  });

  return result;
};

export default async function handler(req, res) {
  const { text } = req.body;

  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }

  try {
    const summary = await summarizeText(text);
    res.status(200).json({ summary });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
