import React from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import Services from './components/Services';
import Footer from './components/Footer';
import './App.css'; // Or './index.css', depending on which one you want to use
import './components/MapSection.js';

function App() {
  return (
    <>
      <Header />
      <Hero />
      <Services />
      <Footer />
    </>
  );
}

export default App;