/**
 * Robust CSV & Excel Signal Parser
 * Handles:
 *  - Single column or multiple columns
 *  - Headers vs. Headerless
 *  - Comma, semicolon, tab, or space delimiters
 *  - NaN, Infinity, blank rows, and text cleaning
 *  - Multi-column machine condition reports (row-based measurements)
 *  - Cells containing parenthetical qualifiers like "0.502(29)" → 0.502
 *  - Cells with dashes "-" or "illegible" → skip
 *  - Excel .xlsx/.xls files (via SheetJS)
 */
import * as XLSX from 'xlsx';


/**
 * Try to parse a messy cell value from machine condition monitoring CSVs.
 * Handles formats like:
 *   "0.502(29.5-"  → 0.502
 *   "0.366(29.5-"  → 0.366
 *   "1.09(60)"     → 1.09
 *   "illeg..."     → NaN
 *   "-"            → NaN
 *   "8.25/1.80"    → 8.25 (take first value)
 */
function parseCell(cell) {
  if (!cell || cell === '' || cell === '-') return NaN;
  const lower = cell.toLowerCase().trim();
  if (lower.startsWith('illeg') || lower === 'n/a' || lower === 'na') return NaN;

  // Extract leading numeric portion before any (, /, [, space
  const match = cell.match(/^-?[\d]*\.?[\d]+/);
  if (match) {
    const val = parseFloat(match[0]);
    return isFinite(val) ? val : NaN;
  }
  return NaN;
}

export function parseVibrationCSV(text, filename = '') {
  if (!text || typeof text !== 'string') {
    throw new Error('Empty or invalid CSV file content.');
  }

  const lines = text.split(/\r\n|\n|\r/).filter(line => line.trim().length > 0);
  if (lines.length === 0) {
    throw new Error('The CSV file is empty.');
  }

  // Detect delimiter (prefer comma, then tab, then semicolon, then whitespace)
  const firstLine = lines[0];
  let delimiter = ',';
  if (firstLine.includes('\t')) delimiter = '\t';
  else if (firstLine.includes(',')) delimiter = ',';
  else if (firstLine.includes(';')) delimiter = ';';
  else delimiter = /\s+/;

  // Split first row into cells
  const firstRowCells = firstLine.split(delimiter).map(c => c.trim().replace(/^["']|["']$/g, ''));

  // Check if first row contains column headers (any non-numeric, non-empty cell → header row)
  const isFirstRowHeader = firstRowCells.some(cell => {
    const clean = cell.replace(/^["']|["']$/g, '').trim();
    return clean !== '' && clean !== '-' && isNaN(parseCell(clean));
  });

  let headers = [];
  let dataStartIndex = 0;

  if (isFirstRowHeader) {
    headers = firstRowCells.map((h, i) => h || `Column_${i + 1}`);
    dataStartIndex = 1;
  } else {
    headers = firstRowCells.map((_, i) => `Column_${i + 1}`);
    dataStartIndex = 0;
  }

  // Collect raw values per column — use lenient parseCell for messy cells
  const columnData = headers.map(() => []);

  for (let i = dataStartIndex; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const cells = line.split(delimiter).map(c => c.trim().replace(/^["']|["']$/g, ''));
    for (let c = 0; c < headers.length; c++) {
      if (c < cells.length) {
        const val = parseCell(cells[c]);
        if (!isNaN(val) && isFinite(val)) {
          columnData[c].push(val);
        }
      }
    }
  }

  // Identify valid numeric signal columns (≥ 10 valid numeric points minimum)
  const numericColumns = [];
  headers.forEach((header, idx) => {
    const samples = columnData[idx];
    if (samples.length >= 10) {
      numericColumns.push({
        columnIndex: idx,
        columnName: header,
        sampleCount: samples.length,
        data: samples,
        preview: samples.slice(0, 10),
        min: Math.min(...samples),
        max: Math.max(...samples),
        mean: samples.reduce((a, b) => a + b, 0) / samples.length
      });
    }
  });

  if (numericColumns.length === 0) {
    throw new Error(
      'No numeric data found in the CSV. ' +
      'Ensure the file contains at least one column of raw vibration samples or numeric measurements.'
    );
  }

  // Identify index/time/metadata columns that should NOT be picked as vibration signals
  const isExcludedColumn = (name) => {
    const n = name.toLowerCase().trim();
    return (
      n.startsWith('sample_no') ||
      n.startsWith('sample_id') ||
      n.startsWith('sampleno') ||
      n.startsWith('sample_num') ||
      n === 'sample_no' ||
      n === 'sample' ||
      n === 'id' ||
      n === 'index' ||
      n === 'row' ||
      n === 'no' ||
      n === 'time' ||
      n.startsWith('time_') ||
      n.includes('timestamp') ||
      n.includes('sampling_rate') ||
      n.includes('defect_size') ||
      n.includes('fault_type') ||
      n.includes('rpm')
    );
  };

  // Rank candidate columns by vibration telemetry relevance
  const scoreColumn = (col) => {
    const n = col.columnName.toLowerCase().trim();
    if (isExcludedColumn(n)) return -1000;

    let score = 0;
    if (n.includes('vibration') && n.includes('acceleration')) score += 1000;
    else if (n.includes('vibration')) score += 800;
    else if (n.includes('acceleration') || n.includes('accel')) score += 700;
    else if (n.includes('de_time') || n.includes('fe_time') || n.includes('ba_time')) score += 600;
    else if (n.includes('de') || n.includes('fe') || n.includes('drive_end')) score += 500;
    else if (n.includes('signal') || n.includes('amplitude') || n.includes('amp')) score += 400;
    else if (n.includes('ch0') || n.includes('ch1') || n === 'x' || n === 'y' || n === 'z') score += 300;
    else if (n.includes('raw') || n.includes('value') || n.includes('data')) score += 100;

    // Realistic vibration signals typically oscillate around 0 with reasonable std dev (< 20 g)
    if (Math.abs(col.mean) < 2.0 && col.max < 50 && col.min > -50) {
      score += 50;
    }
    return score;
  };

  // Pick the column with the highest relevance score
  let defaultColIndex = 0;
  let highestScore = -Infinity;

  numericColumns.forEach((col, idx) => {
    const s = scoreColumn(col);
    if (s > highestScore) {
      highestScore = s;
      defaultColIndex = idx;
    }
  });

  // Check if CSV contains metadata columns (e.g. Sampling_Rate_Hz, Fault_Type, Defect_Size_in)
  let detectedSamplingRate = null;
  let detectedFaultType = null;
  let detectedDefectSize = null;

  headers.forEach((h, idx) => {
    const hLower = h.toLowerCase().trim();
    if (hLower.includes('sampling_rate')) {
      const firstVal = columnData[idx]?.[0];
      if (firstVal && !isNaN(firstVal) && firstVal > 0) {
        detectedSamplingRate = Math.round(firstVal);
      }
    }
    if (hLower.includes('fault_type') || hLower === 'fault' || hLower === 'label') {
      for (let i = dataStartIndex; i < Math.min(lines.length, 20); i++) {
        const cells = lines[i].split(delimiter);
        if (cells[idx]) {
          const val = cells[idx].trim().replace(/^["']|["']$/g, '');
          if (val && isNaN(Number(val))) {
            detectedFaultType = val;
            break;
          }
        }
      }
    }
    if (hLower.includes('defect_size') || hLower.includes('fault_size')) {
      const firstVal = columnData[idx]?.[0];
      if (firstVal && !isNaN(firstVal)) {
        detectedDefectSize = firstVal;
      }
    }
  });

  if (!detectedSamplingRate && filename) {
    const fn = String(filename).toLowerCase();
    if (fn.includes('48k') || fn.includes('48000')) detectedSamplingRate = 48000;
    else if (fn.includes('12k') || fn.includes('12000')) detectedSamplingRate = 12000;
  }

  // ─── Input Type Detection: Raw Vibration vs Frequency Spectrum ─────────────
  let isFrequencySpectrum = false;
  let freqCol = null;
  let ampCol = null;

  if (numericColumns.length >= 2) {
    // 1. Check for monotonic frequency column
    for (let i = 0; i < numericColumns.length; i++) {
      const col = numericColumns[i];
      const name = col.columnName.toLowerCase().trim();
      const data = col.data;
      if (data.length >= 16) {
        const isMonotonic = data[0] >= 0 && data.every((val, idx) => idx === 0 || val > data[idx - 1]);
        if (isMonotonic && (name.includes('freq') || name.includes('hz') || i === 0)) {
          freqCol = col;
          break;
        }
      }
    }

    // 2. If frequency column found, find non-negative amplitude column
    if (freqCol) {
      for (let i = 0; i < numericColumns.length; i++) {
        if (numericColumns[i].columnIndex === freqCol.columnIndex) continue;
        const col = numericColumns[i];
        if (col.data.length === freqCol.data.length) {
          const allNonNeg = col.data.every(val => val >= -1e-6);
          const name = col.columnName.toLowerCase().trim();
          if (allNonNeg && (name.includes('amp') || name.includes('mag') || name.includes('peak') || name.includes('g') || numericColumns.length === 2)) {
            ampCol = col;
            isFrequencySpectrum = true;
            break;
          }
        }
      }
    }
  }

  if (isFrequencySpectrum && freqCol && ampCol) {
    const freqs = freqCol.data;
    const amps = ampCol.data;
    const minFreq = freqs[0];
    const maxFreq = freqs[freqs.length - 1];
    const df = freqs.length > 1 ? (freqs[1] - freqs[0]) : 1.0;

    // Unit detection
    let unit = 'g, peak';
    const ampName = ampCol.columnName.toLowerCase();
    if (ampName.includes('psd') || ampName.includes('g^2') || ampName.includes('g2')) {
      throw new Error(
        'Power Spectral Density (PSD / g²/Hz) is not currently supported. ' +
        'Please export linear acceleration amplitude spectrum (g, peak or g, RMS).'
      );
    }
    if (ampName.includes('db')) {
      throw new Error('Logarithmic (dB) scale spectrum is not supported. Please export linear amplitude spectrum.');
    }
    if (ampName.includes('rms')) unit = 'g, rms';
    else if (ampName.includes('m/s')) unit = 'm/s²';

    // Downsampled spectrum for UI chart rendering
    const step = Math.max(1, Math.floor(freqs.length / 128));
    const chartSpectrum = [];
    for (let k = 0; k < freqs.length; k += step) {
      chartSpectrum.push({
        freqIndex: k,
        freqHz: Math.round(freqs[k] * 10) / 10,
        freq: `${Math.round(freqs[k])} Hz`,
        amplitude: parseFloat(amps[k].toFixed(6))
      });
    }

    return {
      fileType: 'frequency_spectrum',
      input_type: 'frequency_spectrum',
      frequencies: freqs,
      amplitudes: amps,
      spectrumData: chartSpectrum,
      binCount: freqs.length,
      minFreqHz: minFreq,
      maxFreqHz: maxFreq,
      frequencyStepHz: df,
      signalUnit: unit,
      columns: numericColumns,
      defaultColumn: ampCol,
      defaultColumnIndex: ampCol.columnIndex,
      totalDataPoints: amps.length,
      columnCount: numericColumns.length,
      totalRows: lines.length - dataStartIndex,
      detectedSamplingRate: Math.round(maxFreq * 2),
      detectedFaultType,
      detectedDefectSize,
      hasMultiColumnWarning: false
    };
  }

  return {
    fileType: 'raw_vibration',
    input_type: 'raw_vibration',
    columns: numericColumns,
    defaultColumn: numericColumns[defaultColIndex],
    defaultColumnIndex: defaultColIndex,
    totalDataPoints: numericColumns[defaultColIndex]?.data?.length || 0,
    columnCount: numericColumns.length,
    totalRows: lines.length - dataStartIndex,
    detectedSamplingRate,
    detectedFaultType,
    detectedDefectSize,
    hasMultiColumnWarning: (numericColumns[defaultColIndex]?.data?.length || 0) < 1024 && numericColumns.length >= 2
  };
}

export const parseVibrationCsv = parseVibrationCSV;

/**
 * Parse an Excel (.xlsx / .xls) file ArrayBuffer into the same shape as
 * parseVibrationCSV.  Reads the first worksheet, converts it to CSV, then
 * delegates to parseVibrationCSV.
 *
 * @param {ArrayBuffer} buffer  - result of file.arrayBuffer()
 * @param {string}      filename - original filename (for error messages)
 * @returns {object} Same structure as parseVibrationCSV
 */
export function parseVibrationExcel(buffer, filename = 'data.xlsx') {
  try {
    const workbook = XLSX.read(buffer, { type: 'array' });
    const firstSheetName = workbook.SheetNames[0];
    if (!firstSheetName) {
      throw new Error(`No sheets found in the Excel file "${filename}".`);
    }
    const sheet = workbook.Sheets[firstSheetName];
    // Convert the sheet to a CSV string, then reuse the existing CSV parser
    const csvText = XLSX.utils.sheet_to_csv(sheet, { blankrows: false });
    if (!csvText || csvText.trim().length === 0) {
      throw new Error(`The first sheet "${firstSheetName}" appears to be empty.`);
    }
    return parseVibrationCSV(csvText, filename);
  } catch (err) {
    throw new Error(`Excel parse error (${filename}): ${err.message}`);
  }
}
