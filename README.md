# AmbuSmart - Emergency Vehicle Detection Dashboard

A React-based dashboard for emergency vehicle (ambulance) detection with traffic signal automation.

## Features

- **Real-time Distance Tracking**: Monitors ambulance distance as it approaches
- **Acoustic Siren Detection**: Detects emergency vehicle sirens
- **GPS Proximity Validation**: Confirms ambulance location
- **Automated Traffic Control**: Switches traffic lights to green for emergency vehicles
- **System Logging**: Tracks all events with timestamps

## Live Demo

Visit: [https://Vibodharya-Jampale-Sathish.github.io/ambusmart.github.io](https://Vibodharya-Jampale-Sathish.github.io/ambusmart.github.io)

## Local Development

### Prerequisites
- Node.js 14+ installed
- npm or yarn package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Vibodharya-Jampale-Sathish/ambusmart.github.io.git
cd ambusmart.github.io
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

The app will open at `http://localhost:3000`

## Building for Production

Build the optimized production bundle:
```bash
npm run build
```

## Deployment

The project is configured for automatic deployment to GitHub Pages via GitHub Actions.

### Manual Deployment
```bash
npm run deploy
```

This will:
1. Build the React app
2. Deploy to the `gh-pages` branch
3. Publish at the GitHub Pages URL

## Technologies Used

- React 18
- Tailwind CSS
- GitHub Pages
- GitHub Actions

## License

MIT
