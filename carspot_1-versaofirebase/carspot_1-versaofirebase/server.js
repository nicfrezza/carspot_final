require('dotenv').config();
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const cors = require('cors');
const admin = require('firebase-admin');

try {
    if (!admin.apps.length) {
        const serviceAccount = require('C:/Users/nicol/Downloads/carspot/carspot_1-versaofirebase/carspot_1-versaofirebase/yolov8parkingspace-main/yolov8parkingspace-main/serviceAccountKey.json');
        
        admin.initializeApp({
            credential: admin.credential.cert({
                projectId: serviceAccount.project_id,
                clientEmail: serviceAccount.client_email,
                privateKey: serviceAccount.private_key.replace(/\\n/g, '\n')
            }),
            databaseURL: "https://carspot-2f0fa.firebaseio.com"
        });
        
        console.log('Firebase Admin inicializado com sucesso');
    }
} catch (error) {
    console.error('Erro ao inicializar Firebase Admin:', error);
    process.exit(1)
}

const db = admin.firestore();

const app = express();


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

app.get('/', (req, res) => { 
    res.sendFile(path.join(__dirname, 'TelaPrincipal/home.html'));
});

app.use('/FundoImagem', express.static(path.join(__dirname, 'FundoImagem')));


app.get('/inicio', (req, res) => {
    res.sendFile(path.join(__dirname, 'TelaPrincipal/inicio.html')); 
});
app.get('/entrar', (req, res) => {
    res.sendFile(path.join(__dirname, 'Login/sign_in.html'));
});
app.get('/registrar', (req, res) => {
    res.sendFile(path.join(__dirname, 'Login/sign_up.html'));
});
app.get('/recuperar', (req, res) => {
    res.sendFile(path.join(__dirname, 'EsquecerSenha/check_password.html'));
});
app.get('/reset_password', (req, res) => {
    res.sendFile(path.join(__dirname, 'UserPage/esqueciminhasenha.html'));
});
app.get('/pesquisar', (req, res) => {
    res.sendFile(path.join(__dirname, 'TelaPrincipal/index.html'));
});

app.get('/perfil', (req, res) => {
    res.sendFile(path.join(__dirname, 'UserPage/user.html'));
});

app.get('/relatorio', (req, res) => {
    res.sendFile(path.join(__dirname, 'Relatorios/relatorio.html'));
});

app.get('/editar_perfil', (req, res) => {
    res.sendFile(path.join(__dirname, 'UserPage/editar_perfil.html'));
});

app.get('/resetpassword', (req, res) => {
    res.sendFile(path.join(__dirname, 'UserPage/esqueciminhasenha.html'));
});


app.get('/recentes', (req, res) => {
    res.sendFile(path.join(__dirname, 'RecentesPage/recentes.html'));
});

app.get('/fundo', (req, res) => {
    res.sendFile(path.join(__dirname, 'FundoImagem/image.png'));
});

app.get('/logo', (req, res) => {
    res.sendFile(path.join(__dirname, 'FundoImagem/logo12.png'));
});

app.get('/vagas', (req, res) => {
    res.sendFile(path.join(__dirname, 'TelaPrincipal/vagas.html'));
});

app.get('/detalhes', (req, res) => {
    res.sendFile(path.join(__dirname, 'TelaPrincipal/datalhes_vaga.html'));
});


app.get('/coordenadas', (req, res) => {
    res.sendFile(path.join(__dirname, 'TelaPrincipal/coordenadas.html'));
});

app.get('/manifest.json', (req, res) => {
    res.sendFile(path.join(__dirname, '/manifest.json'));
});


app.use(express.static(path.join(__dirname, 'TelaPrincipal')));

app.post('/api/users/register', async (req, res) => {
    console.log('Requisição recebida para registro:', req.body);

    const { nome, email, telefone, senha } = req.body;

    if (!nome || !email || !telefone || !senha) {
        console.log('Campos obrigatórios ausentes');
        return res.status(400).json({ message: 'Todos os campos são obrigatórios.' });
    }

    try {
        try {
            const userExists = await admin.auth().getUserByEmail(email);
            if (userExists) {
                return res.status(400).json({ message: 'Este e-mail já está registrado.' });
            }
        } catch (error) {
            if (error.code !== 'auth/user-not-found') {
                throw error;
            }
        }

        const userRecord = await admin.auth().createUser({
            email,
            password: senha,
            displayName: nome
        });
        
        await db.collection('users').doc(userRecord.uid).set({
            nome,
            telefone,
            email,
            createdAt: admin.firestore.FieldValue.serverTimestamp()
        });

        console.log('Usuário criado com sucesso:', userRecord.uid);
        res.status(201).json({ 
            message: 'Usuário registrado com sucesso',
            userId: userRecord.uid
        });

    } catch (error) {
        console.error('Erro detalhado ao registrar usuário:', error);
        
        let mensagemErro = 'Erro ao registrar usuário.';
        if (error.code === 'auth/email-already-exists') {
            mensagemErro = 'Este e-mail já está registrado.';
        } else if (error.code === 'auth/invalid-email') {
            mensagemErro = 'E-mail inválido.';
        } else if (error.code === 'auth/weak-password') {
            mensagemErro = 'A senha deve ter pelo menos 6 caracteres.';
        }

        res.status(400).json({ 
            message: mensagemErro,
            error: error.message
        });
    }
});

app.post('/api/users/login', async (req, res) => {
  const { email, senha } = req.body;

  if (!email || !senha) {
      return res.status(400).send('Email e senha são obrigatórios.');
  }

  try {
      const userRecord = await admin.auth().getUserByEmail(email);
   
      const idToken = await admin.auth().createCustomToken(userRecord.uid);

      res.status(200).json({ token: idToken });
  } catch (error) {
      console.error('Erro ao fazer login:', error);
      res.status(401).send('Erro ao autenticar usuário: ' + error.message);
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`Servidor rodando na porta ${port}`);
});



